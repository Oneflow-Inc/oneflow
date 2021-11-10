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
#include "OneFlow/OneFlowOps.h"
#include <iostream>
#include <string>
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgTypes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#ifdef WITH_MLIR_CUDA_CODEGEN
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#endif  // WITH_MLIR_CUDA_CODEGEN

using namespace mlir;
using namespace mlir::oneflow;

LogicalResult DumpAssembly(::mlir::PatternRewriter& rewriter, MlirJitOp op) {
  // TODO: now we only need one JIT engine
  auto parent_func_op = op->getParentOfType<FuncOp>();
  auto parent_module_op = parent_func_op->getParentOfType<ModuleOp>();
  SymbolTable symbol_table(parent_module_op);
  std::string mlir;
  llvm::raw_string_ostream os_mlir(mlir);
  symbol_table.lookup(op.op_name())->print(os_mlir);
  op->setAttr("mlir_assembly", rewriter.getStringAttr(mlir));
  return success();
}

// TODO: cfg/multi block support
FuncOp GetOrInsertFuncOp(::mlir::PatternRewriter& rewriter, mlir::Location loc, StringRef func_name,
                         ValueRange operands, ValueRange results, SmallVector<Operation*, 4> ops) {
  BlockAndValueMapping mapping;
  SmallVector<Type, 4> argument_types;
  argument_types.reserve(operands.size());
  SmallVector<Type, 4> result_types;
  argument_types.reserve(results.size());
  for (auto argument : operands) { argument_types.push_back(argument.getType()); }
  for (auto result : results) { result_types.push_back(result.getType()); }
  auto func_type = rewriter.getFunctionType(argument_types, result_types);
  auto first_op = *ops.begin();
  auto parent_func_op = first_op->getParentOfType<FuncOp>();
  auto parent_module_op = parent_func_op->getParentOfType<ModuleOp>();
  SymbolTable symbol_table(parent_module_op);
  OpBuilder::InsertionGuard guard(rewriter);
  Block::iterator insertPt(parent_func_op->getNextNode());
  rewriter.setInsertionPointToStart(&parent_module_op.body().getBlocks().back());
  assert(!parent_func_op->hasAttr("llvm.emit_c_interface"));
  auto function = rewriter.create<mlir::FuncOp>(loc, func_name, func_type);
  function->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(rewriter.getContext()));
  function.body().emplaceBlock();
  function.body().addArguments(argument_types);
  for (auto argument_pair : llvm::zip(operands, function.body().getArguments())) {
    mapping.map(std::get<0>(argument_pair), std::get<1>(argument_pair));
  }
  rewriter.setInsertionPointToStart(&function.body().front());
  ImplicitLocOpBuilder nb(loc, rewriter);
  for (auto op : ops) { nb.clone(*op, mapping); }
  SmallVector<::mlir::Value, 4> mapped_results;
  for (auto result : results) { mapped_results.push_back(mapping.lookup(result)); }
  rewriter.create<ReturnOp>(loc, mapped_results);
  assert(!symbol_table.lookup(func_name));
  return function;
}

NamedAttrList GetJitOpAttributes(::mlir::PatternRewriter& rewriter, StringRef op_name,
                                 int32_t input_size, int32_t output_size,
                                 Operation* op_to_replace) {
  oneflow::UserOpAdaptor op_to_replace_adaptor(op_to_replace->getOperands(),
                                               op_to_replace->getAttrDictionary());
  NamedAttrList attributes;
  attributes.set("op_type_name", rewriter.getStringAttr("mlir_jit"));
  attributes.set("device_tag", op_to_replace_adaptor.device_tag());
  attributes.set("device_name", op_to_replace_adaptor.device_name());
  attributes.set("hierarchy", op_to_replace_adaptor.hierarchy());
  using LBNVec = SmallVector<StringRef, 8>;
  using LBNSegVec = SmallVector<int32_t, 8>;

  LBNVec input_lbn_segment_keys;
  LBNSegVec input_lbn_segment_sizes;
  input_lbn_segment_keys.push_back("in");
  input_lbn_segment_sizes.push_back(input_size);

  attributes.set("input_lbn_segment_keys", rewriter.getStrArrayAttr(input_lbn_segment_keys));
  attributes.set("input_lbn_segment_sizes", rewriter.getI32ArrayAttr(input_lbn_segment_sizes));

  attributes.set("op_name", rewriter.getStringAttr(op_name));

  LBNVec output_lbns;
  LBNVec output_lbn_segment_keys;
  LBNSegVec output_lbn_segment_sizes;
  // TODO: use functions in oneflow to genearated bn
  SmallString<64> output_lbn_storage;
  for (size_t i = 0; i < output_size; i++) {
    output_lbns.push_back(
        (op_name + "/" + "out_" + std::to_string(i)).toStringRef(output_lbn_storage));
    output_lbn_segment_keys.push_back("out");
    output_lbn_segment_sizes.push_back(output_size);
  }
  attributes.set("output_lbns", rewriter.getStrArrayAttr(output_lbns));
  attributes.set("output_lbn_segment_keys", rewriter.getStrArrayAttr(output_lbn_segment_keys));
  attributes.set("output_lbn_segment_sizes", rewriter.getI32ArrayAttr(output_lbn_segment_sizes));
  attributes.set("scope_symbol_id", op_to_replace_adaptor.scope_symbol_id());
  return attributes;
}

::llvm::SmallVector<::mlir::Value, 4> OutlineMulCast(::mlir::PatternRewriter& rewriter,
                                                     mlir::OpResult mul_res,
                                                     mlir::OpResult cast_res) {
  if (llvm::dyn_cast<MlirJitOp>(mul_res.getParentBlock()->getParentOp())) { return {}; }
  if (auto mul_op = llvm::dyn_cast<ScalarMulByTensorOp>(mul_res.getDefiningOp())) {
    if (auto cast_op = llvm::dyn_cast<CastOp>(cast_res.getDefiningOp())) {
      // TODO: extract a function to generate op name for jit op from ops being fused
      SmallString<64> op_name_storage;
      auto op_name =
          (cast_op.op_name() + "__FUSE__" + mul_op.op_name()).toStringRef(op_name_storage);
      SmallVector<::mlir::Value, 2> operands;
      operands.push_back(cast_op.x());
      operands.push_back(mul_op.scalar());
      SmallVector<::mlir::Value, 1> results;
      results.push_back(mul_op.y());
      NamedAttrList attributes =
          GetJitOpAttributes(rewriter, op_name, operands.size(), results.size(), mul_op);
      SmallVector<Operation*, 4> ops = {cast_op, mul_op};
      auto function =
          GetOrInsertFuncOp(rewriter, mul_op->getLoc(), op_name, operands, results, ops);
      auto created = rewriter.create<MlirJitOp>(mul_op.getLoc(), function, attributes, operands);
      assert(DumpAssembly(rewriter, created).succeeded());
      cast_op->dropAllUses();
      cast_op.erase();
      return created->getResults();
    }
  }
  return {};
}

#include "OneFlow/OneFlowPatterns.cpp.inc"

namespace mlir {

namespace oneflow {

void AddLowerToLinalgMemRefPasses(PassManager& pm) {
  pm.addPass(createLowerOneFlowToTosaPass());                     // lower-oneflow-to-tosa
  pm.addPass(createCSEPass());                                    // cse
  pm.addNestedPass<FuncOp>(tosa::createTosaToLinalgOnTensors());  // tosa-to-linalg-on-tensors
  auto p = createLinalgElementwiseOpFusionPass();
  assert(p->initializeOptions("allow-folding-unit-dim-reshapes=true").succeeded());
  pm.addNestedPass<FuncOp>(std::move(p));                     // linalg-fuse-elementwise-ops
  pm.addNestedPass<FuncOp>(createLinalgBufferizePass());      // linalg-bufferize
  pm.addNestedPass<FuncOp>(createTensorBufferizePass());      // tensor-bufferize
  pm.addPass(createTensorConstantBufferizePass());            // tensor-constant-bufferize
  pm.addPass(createFuncBufferizePass());                      // func-bufferize
  pm.addPass(createBufferResultsToOutParamsPass());           // buffer-results-to-out-params
  pm.addPass(createCanonicalizerPass());                      // canonicalize
  pm.addNestedPass<FuncOp>(createFinalizingBufferizePass());  // finalizing-bufferize
}

LogicalResult LowerModuleToLLVM(mlir::MLIRContext* context, ModuleOp module) {
  mlir::PassManager pm(context);
  AddLowerToLinalgMemRefPasses(pm);
  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());  // convert-linalg-to-loops
  pm.addNestedPass<FuncOp>(createLowerToCFGPass());            // convert-scf-to-std
  pm.addPass(createConvertLinalgToLLVMPass());                 // convert-linalg-to-llvm
  pm.addPass(createMemRefToLLVMPass());                        // convert-memref-to-llvm
  pm.addPass(createLowerToLLVMPass());                         // convert-std-to-llvm
  return pm.run(module);
}

#ifdef WITH_MLIR_CUDA_CODEGEN

LogicalResult LowerModuleToCUDALLVM(mlir::MLIRContext* context, ModuleOp module) {
  InitializeLLVMNVPTXBackend();
  mlir::PassManager pm(context);
  AddLowerToLinalgMemRefPasses(pm);
  pm.addNestedPass<FuncOp>(
      createConvertLinalgToParallelLoopsPass());      // convert-linalg-to-parallel-loops
  pm.addNestedPass<FuncOp>(createMapSCFToGPUPass());  // gpu-greedy-parallel-loop-mapping
  pm.addPass(createParallelLoopToGpuPass());          // convert-parallel-loops-to-gpu
  pm.addPass(createGpuKernelOutliningPass());         // gpu-kernel-outlining
  pm.addNestedPass<FuncOp>(createBufferHostRegisterPass());
  pm.addPass(createCanonicalizerPass());  // canonicalize
  pm.addNestedPass<gpu::GPUModuleOp>(createStripDebugInfoPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createLowerAffinePass());
  pm.addNestedPass<gpu::GPUModuleOp>(createLowerGpuOpsToNVVMOpsPass());
  pm.addNestedPass<gpu::GPUModuleOp>(createSerializeToCubinPass());
  pm.addPass(createGpuToLLVMConversionPass());
  return pm.run(module);
}

#endif  // WITH_MLIR_CUDA_CODEGEN

void populateFuserPasses(::mlir::RewritePatternSet& patterns) {
  patterns.add<MulCastPattern>(patterns.getContext());
}

void populateFuserForExistingOp(::mlir::RewritePatternSet& patterns) {
  patterns.add<FusedBiasAddGeluPattern>(patterns.getContext());
  patterns.add<FusedBiasAddDropoutPattern>(patterns.getContext());
  patterns.add<FusedScaleTrilPattern>(patterns.getContext());
  patterns.add<FusedScaleTrilPattern2>(patterns.getContext());
}

}  // namespace oneflow

}  // namespace mlir
