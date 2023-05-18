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
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
