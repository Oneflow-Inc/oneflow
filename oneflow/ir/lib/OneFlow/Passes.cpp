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
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#ifdef WITH_CUDA
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#endif  // WITH_CUDA

using namespace mlir;
using namespace mlir::oneflow;

LogicalResult DumpAssembly(::mlir::PatternRewriter& rewriter, MlirJitOp op) {
  auto context = rewriter.getContext();
  OpBuilder builder(context);
  OwningModuleRef jit_module(
      ModuleOp::create(FileLineColLoc::get(context, "", /*line=*/0, /*column=*/0)));
  auto func_type = rewriter.getFunctionType(op->getOperandTypes(), op->getResultTypes());
  auto function = builder.create<mlir::FuncOp>(op->getLoc(), op.op_name(), func_type);
  function->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(context));
  if (op.body().empty()) {
    op->dump();
    op->emitError("JIT op has an empty body, op name: " + op.op_name());
    return failure();
  }
  rewriter.cloneRegionBefore(op.body(), function.body(), function.body().end());
  jit_module->push_back(function);
  std::string mlir;
  llvm::raw_string_ostream os_mlir(mlir);
  jit_module->print(os_mlir);
  op->setAttr("mlir_assembly", rewriter.getStringAttr(mlir));
  return success();
}

::llvm::SmallVector<::mlir::Value, 4> OutlineFunction(::mlir::PatternRewriter& rewriter,
                                                      mlir::OpResult mul_res,
                                                      mlir::OpResult cast_res) {
  // get matched scale and cast
  // create JIT op and kernel
  if (llvm::dyn_cast<MlirJitOp>(mul_res.getParentBlock()->getParentOp())) { return {}; }
  if (auto mul_op = llvm::dyn_cast<ScalarMulByTensorOp>(mul_res.getDefiningOp())) {
    if (auto cast_op = llvm::dyn_cast<CastOp>(cast_res.getDefiningOp())) {
      NamedAttrList attributes;
      attributes.set("op_type_name", rewriter.getStringAttr("mlir_jit"));
      // TODO: extract a function to strip attrs from an op to be replace
      attributes.set("device_tag", mul_op.device_tagAttr());
      attributes.set("device_name", mul_op.device_nameAttr());
      attributes.set("hierarchy", mul_op.hierarchyAttr());
      using LBNVec = SmallVector<StringRef, 8>;
      using LBNSegVec = SmallVector<int32_t, 8>;

      LBNVec input_lbn_segment_keys;
      LBNSegVec input_lbn_segment_sizes;
      input_lbn_segment_keys.push_back("in");
      input_lbn_segment_sizes.push_back(2);

      attributes.set("input_lbn_segment_keys", rewriter.getStrArrayAttr(input_lbn_segment_keys));
      attributes.set("input_lbn_segment_sizes", rewriter.getI32ArrayAttr(input_lbn_segment_sizes));

      // TODO: extract a function to generate op name for jit op from ops being fused
      SmallString<64> op_name_storage;
      auto op_name =
          (cast_op.op_name() + "__FUSE__" + mul_op.op_name()).toStringRef(op_name_storage);
      attributes.set("op_name", rewriter.getStringAttr(op_name));

      LBNVec output_lbns;
      LBNVec output_lbn_segment_keys;
      LBNSegVec output_lbn_segment_sizes;
      // TODO: use functions in oneflow to genearated bn
      SmallString<64> output_lbn_storage;
      output_lbns.push_back((op_name + "/" + "out_0").toStringRef(output_lbn_storage));
      output_lbn_segment_keys.push_back("out");
      output_lbn_segment_sizes.push_back(1);
      attributes.set("output_lbns", rewriter.getStrArrayAttr(output_lbns));
      attributes.set("output_lbn_segment_keys", rewriter.getStrArrayAttr(output_lbn_segment_keys));
      attributes.set("output_lbn_segment_sizes",
                     rewriter.getI32ArrayAttr(output_lbn_segment_sizes));

      attributes.set("scope_symbol_id", mul_op.scope_symbol_idAttr());
      SmallVector<::mlir::Value, 2> operands;
      operands.push_back(cast_op.x());
      operands.push_back(mul_op.scalar());
      auto created = rewriter.create<MlirJitOp>(mul_op.getLoc(),
                                                /* resultTypes */ mul_op->getResultTypes(),
                                                /* operands */ operands,
                                                /* attributes */ attributes);
      cast_op.replaceAllUsesWith(created);
      created.body().emplaceBlock();
      created.body().addArguments(created->getOperandTypes());
      rewriter.setInsertionPointToStart(&created.body().front());
      auto cast_op_ =
          rewriter.create<CastOp>(cast_op->getLoc(), /* resultTypes */ cast_op->getResultTypes(),
                                  /* operands */ created.body().front().getArguments().take_front(),
                                  /* attributes */ cast_op->getAttrs());
      auto scalar_mul = rewriter.create<ScalarMulByTensorOp>(
          mul_op->getLoc(), /* resultTypes */ mul_op->getResultTypes(),
          /* operands */
          SmallVector<::mlir::Value, 2>({cast_op_.y(), created.body().front().getArgument(1)}),
          /* attributes */ mul_op->getAttrs());
      rewriter.create<ReturnOp>(mul_op->getLoc(), scalar_mul.y());
      assert(DumpAssembly(rewriter, created).succeeded());
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

#ifdef WITH_CUDA

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

#endif  // WITH_CUDA

void populateFuserPasses(::mlir::RewritePatternSet& patterns) {
  patterns.add<OutlineFuseCastScale>(patterns.getContext());
}

}  // namespace oneflow

}  // namespace mlir
