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
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/Passes.h"

#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
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
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#ifdef WITH_MLIR_CUDA_CODEGEN
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"
#endif  // WITH_MLIR_CUDA_CODEGEN

#include "llvm/ADT/STLExtras.h"

#include <iostream>
#include <string>

namespace mlir {

namespace oneflow {

LogicalResult DumpAssembly(::mlir::PatternRewriter& rewriter, MlirJitOp op) {
  // TODO: now we only need one JIT engine
  auto parent_func_op = op->getParentOfType<oneflow::Job>();
  if (!parent_func_op) { return failure(); }
  auto parent_module_op = parent_func_op->getParentOfType<ModuleOp>();
  if (!parent_module_op) { return failure(); }
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
  auto parent_func_op = first_op->getParentOfType<oneflow::Job>();
  if (!parent_func_op) {
    emitError(loc) << "null parent oneflow::Job " << *first_op;
    return nullptr;
  }
  auto parent_module_op = parent_func_op->getParentOfType<ModuleOp>();
  if (!parent_module_op) {
    emitError(loc) << "null ModuleOp " << *first_op;
    return nullptr;
  }
  SymbolTable symbol_table(parent_module_op);
  OpBuilder::InsertionGuard guard(rewriter);
  Block::iterator insertPt(parent_func_op->getNextNode());
  rewriter.setInsertionPointToStart(&parent_module_op.body().getBlocks().back());
  if (parent_func_op->hasAttr("llvm.emit_c_interface")) {
    emitError(loc) << "parent should not has attr of llvm.emit_c_interface " << *parent_func_op;
    return nullptr;
  }
  auto function = rewriter.create<mlir::FuncOp>(loc, func_name, func_type);
  function->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(rewriter.getContext()));
  function.body().emplaceBlock();
  for (auto& arg : argument_types) { function.body().addArguments(arg, loc); }
  for (auto argument_pair : llvm::zip(operands, function.body().getArguments())) {
    mapping.map(std::get<0>(argument_pair), std::get<1>(argument_pair));
  }
  rewriter.setInsertionPointToStart(&function.body().front());
  ImplicitLocOpBuilder nb(loc, rewriter);
  for (auto op : ops) { nb.clone(*op, mapping); }
  SmallVector<::mlir::Value, 4> mapped_results;
  for (auto result : results) { mapped_results.push_back(mapping.lookup(result)); }
  rewriter.create<mlir::ReturnOp>(loc, mapped_results);
  if (symbol_table.lookup(func_name)) {
    emitError(loc) << func_name << " should not be at symbol table of ModuleOp";
    return nullptr;
  }
  return function;
}

NamedAttrList GetJitOpAttributes(::mlir::PatternRewriter& rewriter, StringRef op_name,
                                 int32_t input_size, int32_t output_size,
                                 Operation* op_to_replace) {
  oneflow::UserOpAdaptor op_to_replace_adaptor(op_to_replace->getOperands(),
                                               op_to_replace->getAttrDictionary());
  NamedAttrList attributes;
  attributes.set(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr(),
                 op_to_replace_adaptor.device_tagAttr());
  attributes.set(OpTrait::IsOpConfCompatible<void>::getDeviceNameAttr(),
                 op_to_replace_adaptor.device_name());
  attributes.set(OpTrait::IsOpConfCompatible<void>::getHierarchyAttr(),
                 op_to_replace_adaptor.hierarchyAttr());
  attributes.set(OpTrait::IsOpConfCompatible<void>::getOpNameAttr(),
                 rewriter.getStringAttr(op_name));
  // TODO: use functions in oneflow to genearated bn
  attributes.set(OpTrait::IsOpConfCompatible<void>::getScopeSymbolIDAttr(),
                 op_to_replace_adaptor.scope_symbol_idAttr());
  return attributes;
}

::llvm::SmallVector<::mlir::Value, 4> OutlineBatchMatMul(::mlir::PatternRewriter& rewriter,
                                                         mlir::OpResult matmul_res) {
  if (auto batch_matmul_op = llvm::dyn_cast<BatchMatmulOp>(matmul_res.getDefiningOp())) {
    auto op_name = batch_matmul_op.op_name();
    SmallVector<::mlir::Value, 2> operands;
    operands.push_back(batch_matmul_op.a());
    operands.push_back(batch_matmul_op.b());
    SmallVector<::mlir::Value, 1> results;
    results.push_back(batch_matmul_op.out());
    NamedAttrList attributes =
        GetJitOpAttributes(rewriter, op_name, operands.size(), results.size(), batch_matmul_op);
    SmallVector<Operation*, 4> ops = {batch_matmul_op};
    auto function =
        GetOrInsertFuncOp(rewriter, batch_matmul_op->getLoc(), op_name, operands, results, ops);
    auto created =
        rewriter.create<MlirJitOp>(batch_matmul_op.getLoc(), function, attributes, operands);
    if (failed(DumpAssembly(rewriter, created))) exit(1);
    return created->getResults();
  }
  return {};
}

static StringRef sanitizeIdentifier(StringRef name, SmallString<16>& buffer,
                                    StringRef allowedPunctChars = "$._",
                                    bool allowTrailingDigit = true) {
  assert(!name.empty() && "Shouldn't have an empty name here");

  auto copyNameToBuffer = [&] {
    for (char ch : name) {
      if (llvm::isAlnum(ch) || allowedPunctChars.contains(ch))
        buffer.push_back(ch);
      else if (ch == ' ')
        buffer.push_back('_');
      else
        buffer.append(llvm::utohexstr((unsigned char)ch));
    }
  };

  // Check to see if this name is valid. If it starts with a digit, then it
  // could conflict with the autogenerated numeric ID's, so add an underscore
  // prefix to avoid problems.
  if (isdigit(name[0])) {
    buffer.push_back('_');
    copyNameToBuffer();
    return buffer;
  }

  // If the name ends with a trailing digit, add a '_' to avoid potential
  // conflicts with autogenerated ID's.
  if (!allowTrailingDigit && isdigit(name.back())) {
    copyNameToBuffer();
    buffer.push_back('_');
    return buffer;
  }

  // Check to see that the name consists of only valid identifier characters.
  for (char ch : name) {
    if (!llvm::isAlnum(ch) && !allowedPunctChars.contains(ch)) {
      copyNameToBuffer();
      return buffer;
    }
  }

  // If there are no invalid characters, return the original name.
  return name;
}

::llvm::SmallVector<::mlir::Value, 4> OutlineMulCast(::mlir::PatternRewriter& rewriter,
                                                     mlir::OpResult mul_res,
                                                     mlir::OpResult cast_res) {
  auto mul_op = mul_res.getDefiningOp();
  auto scale = mlir::Value();
  auto output = mlir::Value();
  if (auto scalar_mul_op = llvm::dyn_cast<ScalarMulByTensorOp>(mul_op)) {
    scale = scalar_mul_op.scalar();
    output = scalar_mul_op.y();
  } else if (auto broadcast_mul_op = llvm::dyn_cast<BroadcastMulOp>(mul_op)) {
    scale = broadcast_mul_op.y();
    output = broadcast_mul_op.z();
  } else {
    mul_res.getDefiningOp()->emitError("pattern mul(cast(x), scalar) doesn't support this op");
    exit(1);
  }
  if (!mul_op->hasTrait<OpTrait::IsOpConfCompatible>()) {
    mul_res.getDefiningOp()->emitError("not OpConf compatible");
    exit(1);
  }
  if (auto cast_op = llvm::dyn_cast<CastOp>(cast_res.getDefiningOp())) {
    // TODO: extract a function to generate op name for jit op from ops being fused
    SmallString<64> op_name_storage;
    auto op_name =
        (cast_op.op_name() + "__FUSE__"
         + mul_op->getAttrOfType<StringAttr>(OpTrait::IsOpConfCompatible<void>::getOpNameAttr())
               .getValue()
               .str())
            .toStringRef(op_name_storage);
    SmallString<16> tempBuffer;
    op_name = sanitizeIdentifier(op_name, tempBuffer);
    SmallVector<::mlir::Value, 2> operands;
    operands.push_back(cast_op.in());
    operands.push_back(scale);
    SmallVector<::mlir::Value, 1> results;
    results.push_back(output);
    NamedAttrList attributes =
        GetJitOpAttributes(rewriter, op_name, operands.size(), results.size(), mul_op);
    SmallVector<Operation*, 4> ops = {cast_op, mul_op};
    auto function = GetOrInsertFuncOp(rewriter, mul_op->getLoc(), op_name, operands, results, ops);
    auto created = rewriter.create<MlirJitOp>(mul_op->getLoc(), function, attributes, operands);
    if (failed(DumpAssembly(rewriter, created))) { exit(1); }
    cast_op->dropAllUses();
    cast_op.erase();
    return created->getResults();
  }
  return {};
}

::llvm::SmallVector<::mlir::Value, 4> CreateGPUMemcpyOpFromMemrefCopy(
    ::mlir::PatternRewriter& rewriter, ::mlir::memref::CopyOp copyOp) {
  // NOTE: to get lowered to LLVM, it has to be async
  ::mlir::ValueRange empty_async_dependencies{};
  auto token = rewriter.getType<gpu::AsyncTokenType>();
  auto t0 =
      rewriter.create<gpu::WaitOp>(copyOp->getLoc(), token, empty_async_dependencies).asyncToken();
  auto t2 = rewriter
                .create<gpu::MemcpyOp>(copyOp->getLoc(),
                                       /*optional asyncToken*/ token,
                                       /*asyncDependencies*/ llvm::SmallVector<Value, 1>({t0}),
                                       /*dst*/ copyOp.target(),
                                       /*src*/ copyOp.source())
                .getResults();
  rewriter.create<gpu::WaitOp>(copyOp->getLoc(), llvm::None, t2);
  return {};
}

bool IsScalarTensor(Value value) {
  if (auto tensor = value.getType().dyn_cast<RankedTensorType>()) {
    return tensor.getNumElements() == 1;
  }
  return false;
}

}  // namespace oneflow

}  // namespace mlir

#include "OneFlow/OneFlowPatterns.cpp.inc"

namespace mlir {

namespace oneflow {

void BroadcastMulOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
  results.insert<BroadcastMulToScalarMulPattern>(context);
}

void AddLowerToLinalgMemRefPasses(PassManager& pm) {
  pm.addPass(createLowerOneFlowToTosaPass());            // lower-oneflow-to-tosa
  pm.addPass(createCSEPass());                           // cse
  pm.addNestedPass<FuncOp>(tosa::createTosaToLinalg());  // tosa-to-linalg-on-tensors
  auto p = createLinalgElementwiseOpFusionPass();
  if (p->initializeOptions("allow-folding-unit-dim-reshapes=true").failed()) exit(1);
  pm.addNestedPass<FuncOp>(std::move(p));                           // linalg-fuse-elementwise-ops
  pm.addNestedPass<FuncOp>(createLinalgBufferizePass());            // linalg-bufferize
  pm.addNestedPass<FuncOp>(createTensorBufferizePass());            // tensor-bufferize
  pm.addPass(createFuncBufferizePass());                            // func-bufferize
  pm.addPass(bufferization::createBufferResultsToOutParamsPass());  // buffer-results-to-out-params
  pm.addPass(createCanonicalizerPass());                            // canonicalize
  pm.addNestedPass<FuncOp>(
      mlir::bufferization::createFinalizingBufferizePass());  // finalizing-bufferize
}

LogicalResult LowerModuleToLLVM(mlir::MLIRContext* context, ModuleOp module) {
  mlir::PassManager pm(context);
  AddLowerToLinalgMemRefPasses(pm);
  pm.addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());  // convert-linalg-to-loops
  pm.addNestedPass<FuncOp>(createConvertSCFToCFPass());        // convert-scf-to-cf
  pm.addPass(createConvertLinalgToLLVMPass());                 // convert-linalg-to-llvm
  pm.addPass(createMemRefToLLVMPass());                        // convert-memref-to-llvm
  pm.addPass(createLowerToLLVMPass());                         // convert-std-to-llvm
  pm.addPass(createReconcileUnrealizedCastsPass());            // reconcile-unrealized-casts
  return pm.run(module);
}

#ifdef WITH_MLIR_CUDA_CODEGEN

LogicalResult LowerModuleToCUDALLVM(mlir::MLIRContext* context, ModuleOp module) {
  InitializeLLVMNVPTXBackend();
  mlir::PassManager pm(context);
  bool enable_ir_printing =
      ::oneflow::ParseBooleanFromEnv("ONEFLOW_MLIR_ENABLE_IR_PRINTING", false);
  context->disableMultithreading(enable_ir_printing);
  AddLowerToLinalgMemRefPasses(pm);
  pm.addNestedPass<FuncOp>(
      createConvertLinalgToParallelLoopsPass());             // convert-linalg-to-parallel-loops
  pm.addPass(createMapSCFToGPUPass());                       // gpu-greedy-parallel-loop-mapping
  pm.addPass(createParallelLoopToGpuPass());                 // convert-parallel-loops-to-gpu
  pm.addPass(createGpuKernelOutliningPass());                // gpu-kernel-outlining
  pm.addNestedPass<FuncOp>(createBufferHostRegisterPass());  // buffer-host-register
  pm.addPass(createCanonicalizerPass());                     // canonicalize
  // -pass-pipeline='gpu.module([PASS1][PASS2]...)'
  pm.addNestedPass<gpu::GPUModuleOp>(createStripDebugInfoPass());        // strip-debuginfo
  pm.addNestedPass<gpu::GPUModuleOp>(createLowerAffinePass());           // lower-affine
  pm.addNestedPass<gpu::GPUModuleOp>(createLowerGpuOpsToNVVMOpsPass());  // convert-gpu-to-nvvm
  pm.addNestedPass<gpu::GPUModuleOp>(createSerializeToCubinPass());      // out-of-tree-gpu-to-cubin
  pm.addNestedPass<FuncOp>(createGpuCopyArgPass());                      // buffer-host-register
  pm.addPass(createGpuToLLVMConversionPass());
  if (enable_ir_printing) pm.enableIRPrinting();
  return pm.run(module);
}

#endif  // WITH_MLIR_CUDA_CODEGEN

void populateFuserPasses(::mlir::RewritePatternSet& patterns) {
  patterns.add<MulCastPattern>(patterns.getContext());
  patterns.add<BatchMatmulPattern>(patterns.getContext());
}

void populateFuserForExistingOp(::mlir::RewritePatternSet& patterns) {
  patterns.add<FusedBiasAddGeluPattern>(patterns.getContext());
  patterns.add<FusedScaleTrilPattern>(patterns.getContext());
  patterns.add<FusedScaleTrilPattern2>(patterns.getContext());
  patterns.add<NormalizationAddReluPattern>(patterns.getContext());
}

void populateGpuHelperPatterns(::mlir::RewritePatternSet& patterns) {
  patterns.add<ReplaceCopyWithGPUPattern>(patterns.getContext());
}

}  // namespace oneflow

}  // namespace mlir
