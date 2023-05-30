/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "OneFlow/Transform/OneFlowMemPool.h"
#include "OneFlow/OneFlowPDLLPatterns.h"
#include "Transform/TransformDialectExtension.h"
#include "Transform/TransformStateExtension.h"
#include "mlir/Conversion/VectorToGPU/VectorToGPU.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::oneflow;
using namespace mlir::transform;

namespace {
struct MemrefCopyOpFoldPatterns final : public OpRewritePattern<memref::CopyOp> {
 public:
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::CopyOp op, PatternRewriter& rewriter) const override {
    if (op.getSource() == op.getTarget()) rewriter.eraseOp(op);
    return success();
  }
};

std::optional<SmallVector<int64_t>> gpuMmaUnrollOrder(vector::ContractionOp contract) {
  SmallVector<int64_t> order;
  // First make reduction the outer dimensions.
  for (auto [index, iter] : llvm::enumerate(contract.getIteratorTypes())) {
    if (vector::isReductionIterator(iter)) { order.push_back(index); }
  }

  llvm::SmallDenseSet<int64_t> dims;
  for (AffineExpr expr : contract.getIndexingMapsArray()[0].getResults()) {
    dims.insert(expr.cast<AffineDimExpr>().getPosition());
  }
  // Then parallel dimensions that are part of Lhs as we want to re-use Lhs.
  for (auto [index, iter] : llvm::enumerate(contract.getIteratorTypes())) {
    if (vector::isParallelIterator(iter) && dims.count(index)) { order.push_back(index); }
  }
  // Then the remaining parallel loops.
  for (auto [index, iter] : llvm::enumerate(contract.getIteratorTypes())) {
    if (vector::isParallelIterator(iter) && !dims.count(index)) { order.push_back(index); }
  }
  return order;
}

std::optional<SmallVector<int64_t>> getWmmaNativeVectorSize(Operation* op) {
  // Currently hardcode the size of wmma operation. When more cases are
  // supported this should be picked based on what the backend supports.
  int64_t m = 16;
  int64_t n = 16;
  if (auto contract = dyn_cast<vector::ContractionOp>(op)) {
    int64_t k = contract.getLhsType().getElementType().isF16() ? 16 : 8;
    SmallVector<int64_t> nativeSize(contract.getIteratorTypes().size() - 3, 1);
    nativeSize.append({m, n, k});
    return nativeSize;
  }
  if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
    SmallVector<int64_t> nativeSize(writeOp.getVectorType().getRank() - 2, 1);
    nativeSize.append({m, n});
    return nativeSize;
  }
  if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
    // Transfer read ops may need different shapes based on how they are being
    // used. For simplicity just match the shape used by the extract strided op.
    VectorType sliceType;
    for (Operation* users : op->getUsers()) {
      auto extract = dyn_cast<vector::ExtractStridedSliceOp>(users);
      if (!extract) return std::nullopt;
      auto vecType = extract.getResult().getType().cast<VectorType>();
      if (sliceType && sliceType != vecType) return std::nullopt;
      sliceType = vecType;
    }
    return llvm::to_vector(sliceType.getShape());
  }
  if ((OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1)) {
    if (auto vecType = op->getResultTypes()[0].dyn_cast<VectorType>()) {
      // TODO: The condition for unrolling elementwise should be restricted
      // only to operations that need unrolling (connected to the contract).
      if (vecType.getRank() < 2) return std::nullopt;

      // First check whether there is a slice to infer the shape from. This is
      // required for cases where the accumulator type differs from the input
      // types, in which case we will see an `arith.ext_` between the contract
      // and transfer_read which needs to be unrolled.
      VectorType sliceType;
      for (Operation* users : op->getUsers()) {
        auto extract = dyn_cast<vector::ExtractStridedSliceOp>(users);
        if (!extract) return std::nullopt;
        auto vecType = extract.getResult().getType().cast<VectorType>();
        if (sliceType && sliceType != vecType) return std::nullopt;
        sliceType = vecType;
      }
      if (sliceType) return llvm::to_vector(sliceType.getShape());

      // Else unroll for trailing elementwise.
      SmallVector<int64_t> nativeSize(vecType.getRank() - 2, 1);
      // Map elementwise ops to the output shape.
      nativeSize.append({m, n});
      return nativeSize;
    }
  }
  return std::nullopt;
}

}  // namespace

DiagnosedSilenceableFailure transform_dialect::EliminateCopyOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  MLIRContext* ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<MemrefCopyOpFoldPatterns>(patterns.getContext());
  mlir::oneflow::populateAllocEliminationPatterns(patterns);
  SmallVector<Operation*> ops;
  GreedyRewriteConfig config;
  target->walk([&](Operation* nestedOp) {
    if (target != nestedOp) ops.push_back(nestedOp);
  });
  LogicalResult result = applyOpPatternsAndFold(ops, std::move(patterns), config);
  if (failed(result)) { return DiagnosedSilenceableFailure::definiteFailure(); }
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure transform_dialect::ExplicitLinalgOutcomeOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  MLIRContext* ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  linalg::populateFoldUnitExtentDimsViaSlicesPatterns(patterns);
  SmallVector<Operation*> ops;
  GreedyRewriteConfig config;
  target->walk([&](Operation* nestedOp) {
    if (target != nestedOp) ops.push_back(nestedOp);
  });
  LogicalResult result = applyOpPatternsAndFold(ops, std::move(patterns), config);
  if (failed(result)) { return DiagnosedSilenceableFailure::definiteFailure(); }
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure transform_dialect::CanonicalizationOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  MLIRContext* ctx = target->getContext();
  RewritePatternSet patterns(ctx);
  for (Dialect* dialect : ctx->getLoadedDialects()) dialect->getCanonicalizationPatterns(patterns);
  for (RegisteredOperationName op : ctx->getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, ctx);
  SmallVector<Operation*> ops;
  GreedyRewriteConfig config;
  target->walk([&](Operation* nestedOp) {
    if (target != nestedOp) ops.push_back(nestedOp);
  });
  LogicalResult result = applyOpPatternsAndFold(ops, std::move(patterns), config);
  if (failed(result)) { return DiagnosedSilenceableFailure::definiteFailure(); }
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure transform_dialect::FoldAllocOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  if (auto func = llvm::dyn_cast<func::FuncOp>(target)) { applyFoldAlloc(func); }
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure transform_dialect::ResultsToOutParamsOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  if (auto module = llvm::dyn_cast<ModuleOp>(target)) {
    if (failed(bufferization::promoteBufferResultsToOutParams(module, {}))) {
      return DiagnosedSilenceableFailure::definiteFailure();
    }
  }
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure transform_dialect::VectorToMMAOp::applyToOne(
    Operation* target, transform::ApplyToEachResultList& results,
    transform::TransformState& state) {
  if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
    target->emitOpError("applies only to isolated-from-above targets because it "
                        "needs to apply "
                        "patterns greedily");
    return emitDefaultDefiniteFailure(target);
  }

  auto funcOp = dyn_cast<func::FuncOp>(target);
  if (!funcOp) {
    target->emitOpError("Must apply to a func op");
    return emitDefaultDefiniteFailure(target);
  }

  if (!(getUseMmaSync() ^ getUseWmma())) {
    target->emitOpError("Exactly one of use_mma_sync or use_wmma must be specified");
    return emitDefaultDefiniteFailure(target);
  }

  MLIRContext* ctx = target->getContext();
  GreedyRewriteConfig config;

  if (getUseWmma()) {
    RewritePatternSet patterns(ctx);
    auto unrollOrder = [](Operation* op) -> std::optional<SmallVector<int64_t>> {
      auto contract = dyn_cast<vector::ContractionOp>(op);
      if (!contract) return std::nullopt;
      return gpuMmaUnrollOrder(contract);
    };
    vector::populateVectorUnrollPatterns(patterns, vector::UnrollVectorOptions()
                                                       .setNativeShapeFn(getWmmaNativeVectorSize)
                                                       .setUnrollTraversalOrderFn(unrollOrder));

    if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns), config))) {
      target->emitOpError("unroll vector for wmma failed to apply");
      return emitDefaultDefiniteFailure(target);
    }
  }

  {
    RewritePatternSet patterns(ctx);
    mlir::vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);
    populatePrepareVectorToMMAPatterns(patterns, getUseMmaSync());
    if (failed(applyPatternsAndFoldGreedily(target, std::move(patterns), config))) {
      target->emitOpError("vector to mma preparation patterns failed to apply");
      return emitDefaultDefiniteFailure(target);
    }
  }

  IRRewriter rewriter(target->getContext());
  auto ret = DiagnosedSilenceableFailure::success();
  if (getUseWmma()) {
    if (failed(convertVectorToMMAOps(rewriter, target)))
      ret = emitDefiniteFailure("vector to wmma patterns failed to apply");
    return ret;
  }

  if (failed(convertVectorToNVVMCompatibleMMASync(rewriter, funcOp))) {
    target->emitOpError("vector to mma patterns failed to apply");
    return emitDefaultDefiniteFailure(target);
  }
  // Using TF32 for Float.
  RewritePatternSet f32ToTF32patterns(funcOp.getContext());
  nvgpu::populateMmaSyncF32ToTF32Patterns(f32ToTF32patterns, nvgpu::MmaSyncF32Lowering::TF32);
  if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(f32ToTF32patterns), config))) {
    target->emitOpError("vector to mma F32ToTF32 patterns failed to apply");
    return emitDefaultDefiniteFailure(target);
  }

  return ret;
}

DiagnosedSilenceableFailure transform_dialect::CSEOp::applyToOne(Operation* target,
                                                                 ApplyToEachResultList& results,
                                                                 transform::TransformState& state) {
  auto context = target->getContext();
  mlir::PassManager pm(context);
  pm.addPass(createCSEPass());
  if (failed(pm.run(target))) return mlir::emitDefiniteFailure(target, "greedy patterns failed");
  return DiagnosedSilenceableFailure::success();
}

namespace {
class OneFlowTransformDialectExtension
    : public transform::TransformDialectExtension<OneFlowTransformDialectExtension> {
 public:
  using Base::Base;

  void init() {
    declareDependentDialect<pdl::PDLDialect>();
    registerTransformOps<
#define GET_OP_LIST
#include "Transform/TransformDialectExtension.cpp.inc"
        >();
    registerTypes<
#define GET_TYPEDEF_LIST
#include "Transform/TransformDialectExtensionTypes.cpp.inc"
        >();
  }
};
}  // namespace

// These are automatically generated by ODS but are not used as the Transform
// dialect uses a different dispatch mechanism to support dialect extensions.
LLVM_ATTRIBUTE_UNUSED static OptionalParseResult generatedTypeParser(AsmParser& parser,
                                                                     StringRef* mnemonic,
                                                                     Type& value);
LLVM_ATTRIBUTE_UNUSED static LogicalResult generatedTypePrinter(Type def, AsmPrinter& printer);

#define GET_TYPEDEF_CLASSES
#include "Transform/TransformDialectExtensionTypes.cpp.inc"

#define GET_OP_CLASSES
#include "Transform/TransformDialectExtension.cpp.inc"

void mlir::oneflow::transform_dialect::registerTransformDialectExtension(
    DialectRegistry& registry) {
  registry.addExtensions<OneFlowTransformDialectExtension>();
}
