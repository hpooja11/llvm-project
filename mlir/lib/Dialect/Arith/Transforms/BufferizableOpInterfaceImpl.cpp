//===- BufferizableOpInterfaceImpl.cpp - Impl. of BufferizableOpInterface -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {
/// Bufferization of arith.constant. Replace with memref.get_global.
struct ConstantOpInterface
    : public BufferizableOpInterface::ExternalModel<ConstantOpInterface,
                                                    arith::ConstantOp> {
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto constantOp = cast<arith::ConstantOp>(op);
    auto type = dyn_cast<RankedTensorType>(constantOp.getType());

    // Only ranked tensors are supported.
    if (!type)
      return failure();

    Attribute memorySpace;
    if (auto memSpace = options.defaultMemorySpaceFn(type))
      memorySpace = *memSpace;
    else
      return constantOp->emitError("could not infer memory space");

    // Only constants inside a module are supported.
    auto moduleOp = constantOp->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return failure();

    // Create global memory segment and replace tensor with memref pointing to
    // that memory segment.
    FailureOr<memref::GlobalOp> globalOp =
        getGlobalFor(constantOp, options.bufferAlignment, memorySpace);
    if (failed(globalOp))
      return failure();
    memref::GlobalOp globalMemref = *globalOp;
    replaceOpWithNewBufferizedOp<memref::GetGlobalOp>(
        rewriter, op, globalMemref.getType(), globalMemref.getName());

    return success();
  }

  bool isWritable(Operation *op, Value value,
                  const AnalysisState &state) const {
    // Memory locations returned by memref::GetGlobalOp may not be written to.
    assert(isa<OpResult>(value));
    return false;
  }
};

struct IndexCastOpInterface
    : public BufferizableOpInterface::ExternalModel<IndexCastOpInterface,
                                                    arith::IndexCastOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {{op->getResult(0), BufferRelation::Equivalent}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto castOp = cast<arith::IndexCastOp>(op);
    auto resultTensorType = cast<TensorType>(castOp.getType());

    FailureOr<Value> source = getBuffer(rewriter, castOp.getIn(), options);
    if (failed(source))
      return failure();
    auto sourceType = cast<BaseMemRefType>(source->getType());
    auto shape = sourceType.getShape();
    int64_t rank = shape.size();

    if (shape.size() > 0) {
      // Create a memref type for the result
      auto resultElementType = resultTensorType.getElementType();
      SmallVector<int64_t, 4> resultShape(shape.begin(), shape.end());
      auto resultMemRef = MemRefType::get(shape, resultElementType);

      // Handling dynamic dimensions
      SmallVector<Value, 4> dynamicSizes;
      for (int64_t dim = 0; dim < rank; ++dim) {
        if (shape[dim] == ShapedType::kDynamic) {
          // Get dynamic size of the dimension
          Value dimSize =
              rewriter.create<memref::DimOp>(op->getLoc(), *source, dim);
          dynamicSizes.push_back(dimSize);
        }
      }

      Value alloc;
      if (dynamicSizes.empty())
        alloc = rewriter.create<memref::AllocOp>(op->getLoc(), resultMemRef);
      else
        alloc = rewriter.create<memref::AllocOp>(op->getLoc(), resultMemRef,
                                                 dynamicSizes);

      // Constant for loop bounds
      SmallVector<Value, 4> lowerBounds(
          rank, rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0));
      SmallVector<Value, 4> upperBounds;
      SmallVector<Value, 4> steps(
          rank, rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1));

      for (int64_t dim = 0; dim < rank; ++dim) {
        if (shape[dim] == ShapedType::kDynamic) {
          Value dimSize =
              rewriter.create<memref::DimOp>(op->getLoc(), *source, dim);
          upperBounds.push_back(dimSize);
        } else {
          upperBounds.push_back(rewriter.create<arith::ConstantIndexOp>(
              op->getLoc(), shape[dim]));
        }
      }

      // Function to create nested loops
      std::function<void(OpBuilder &, Location, SmallVector<Value, 4> &,
                         int64_t)>
          createNestedLoops;
      createNestedLoops = [&](OpBuilder &nestedBuilder, Location loc,
                              SmallVector<Value, 4> &indices, int64_t depth) {
        if (depth == rank) {
          // Base case: all dimensions are handled
          Value elem =
              nestedBuilder.create<memref::LoadOp>(loc, *source, indices);

          // Cast the element to the result type
          Value casted = nestedBuilder.create<arith::IndexCastOp>(
              loc, resultTensorType.getElementType(), elem);

          // Store the casted element into the result memref
          nestedBuilder.create<memref::StoreOp>(loc, casted, alloc, indices);
        } else {
          Value lb = lowerBounds[depth];
          Value ub = upperBounds[depth];
          Value step = steps[depth];
          nestedBuilder.create<scf::ForOp>(
              loc, lb, ub, step, ValueRange{},
              [&](OpBuilder &innerBuilder, Location innerLoc, Value iv,
                  ValueRange) {
                indices.push_back(iv);
                createNestedLoops(innerBuilder, innerLoc, indices, depth + 1);
                indices.pop_back();
                innerBuilder.create<scf::YieldOp>(innerLoc);
              });
        }
      };
      SmallVector<Value, 4> indices;
      createNestedLoops(rewriter, op->getLoc(), indices, 0);
      replaceOpWithNewBufferizedOp<arith::IndexCastOp>(rewriter, op,
                                                       resultMemRef, alloc);
    } else {
      // Result type should have same layout and address space as the source
      // type.
      BaseMemRefType resultType;
      if (auto rankedMemRefType = dyn_cast<MemRefType>(sourceType)) {
        resultType = MemRefType::get(
            rankedMemRefType.getShape(), resultTensorType.getElementType(),
            rankedMemRefType.getLayout(), rankedMemRefType.getMemorySpace());
      } else {
        auto unrankedMemrefType = cast<UnrankedMemRefType>(sourceType);
        resultType =
            UnrankedMemRefType::get(resultTensorType.getElementType(),
                                    unrankedMemrefType.getMemorySpace());
      }
      replaceOpWithNewBufferizedOp<arith::IndexCastOp>(rewriter, op, resultType,
                                                       *source);
    }
    return success();
  }
};

/// Bufferization of arith.select. Just replace the operands.
struct SelectOpInterface
    : public BufferizableOpInterface::ExternalModel<SelectOpInterface,
                                                    arith::SelectOp> {
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return false;
  }

  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {{op->getOpResult(0) /*result*/, BufferRelation::Equivalent,
             /*isDefinite=*/false}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto selectOp = cast<arith::SelectOp>(op);
    Location loc = selectOp.getLoc();

    // Elementwise conditions are not supported yet. To bufferize such an op,
    // it could be lowered to an elementwise "linalg.generic" with a new
    // "tensor.empty" out tensor, followed by "empty tensor elimination". Such
    // IR will bufferize.
    if (!selectOp.getCondition().getType().isInteger(1))
      return op->emitOpError("only i1 condition values are supported");

    // TODO: It would be more efficient to copy the result of the `select` op
    // instead of its OpOperands. In the worst case, 2 copies are inserted at
    // the moment (one for each tensor). When copying the op result, only one
    // copy would be needed.
    FailureOr<Value> maybeTrueBuffer =
        getBuffer(rewriter, selectOp.getTrueValue(), options);
    FailureOr<Value> maybeFalseBuffer =
        getBuffer(rewriter, selectOp.getFalseValue(), options);
    if (failed(maybeTrueBuffer) || failed(maybeFalseBuffer))
      return failure();
    Value trueBuffer = *maybeTrueBuffer;
    Value falseBuffer = *maybeFalseBuffer;

    // The "true" and the "false" operands must have the same type. If the
    // buffers have different types, they differ only in their layout map. Cast
    // both of them to the most dynamic MemRef type.
    if (trueBuffer.getType() != falseBuffer.getType()) {
      auto targetType =
          bufferization::getBufferType(selectOp.getResult(), options);
      if (failed(targetType))
        return failure();
      if (trueBuffer.getType() != *targetType)
        trueBuffer =
            rewriter.create<memref::CastOp>(loc, *targetType, trueBuffer);
      if (falseBuffer.getType() != *targetType)
        falseBuffer =
            rewriter.create<memref::CastOp>(loc, *targetType, falseBuffer);
    }

    replaceOpWithNewBufferizedOp<arith::SelectOp>(
        rewriter, op, selectOp.getCondition(), trueBuffer, falseBuffer);
    return success();
  }

  FailureOr<BaseMemRefType>
  getBufferType(Operation *op, Value value, const BufferizationOptions &options,
                SmallVector<Value> &invocationStack) const {
    auto selectOp = cast<arith::SelectOp>(op);
    assert(value == selectOp.getResult() && "invalid value");
    auto trueType = bufferization::getBufferType(selectOp.getTrueValue(),
                                                 options, invocationStack);
    auto falseType = bufferization::getBufferType(selectOp.getFalseValue(),
                                                  options, invocationStack);
    if (failed(trueType) || failed(falseType))
      return failure();
    if (*trueType == *falseType)
      return *trueType;
    if (trueType->getMemorySpace() != falseType->getMemorySpace())
      return op->emitError("inconsistent memory space on true/false operands");

    // If the buffers have different types, they differ only in their layout
    // map.
    auto memrefType = llvm::cast<MemRefType>(*trueType);
    return getMemRefTypeWithFullyDynamicLayout(
        RankedTensorType::get(memrefType.getShape(),
                              memrefType.getElementType()),
        memrefType.getMemorySpace());
  }
};

} // namespace

void mlir::arith::registerBufferizableOpInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, ArithDialect *dialect) {
    ConstantOp::attachInterface<ConstantOpInterface>(*ctx);
    IndexCastOp::attachInterface<IndexCastOpInterface>(*ctx);
    SelectOp::attachInterface<SelectOpInterface>(*ctx);
  });
}
