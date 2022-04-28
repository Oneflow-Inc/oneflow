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
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OperationSupport.h"
#include "oneflow/core/framework/random_generator.h"

#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"

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
  if (auto found = symbol_table.lookup(op.op_name())) {
    found->print(os_mlir);
  } else {
    parent_module_op->dump();
    return op.emitError("symbol of jit function not found: " + op.op_name());
  }
  op->setAttr("mlir_assembly", rewriter.getStringAttr(mlir));
  return success();
}

// TODO: cfg/multi block support
func::FuncOp GetOrInsertFuncOp(::mlir::PatternRewriter& rewriter, mlir::Location loc,
                               StringRef func_name, ValueRange operands, ValueRange results,
                               SmallVector<Operation*, 4> ops) {
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
  rewriter.setInsertionPointToStart(parent_module_op.getBody());
  if (parent_func_op->hasAttr("llvm.emit_c_interface")) {
    emitError(loc) << "parent should not has attr of llvm.emit_c_interface " << *parent_func_op;
    return nullptr;
  }
  auto function = rewriter.create<func::FuncOp>(loc, func_name, func_type);
  function->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(rewriter.getContext()));
  function.getBody().emplaceBlock();
  for (auto& arg : argument_types) { function.getBody().addArguments(arg, loc); }
  for (auto argument_pair : llvm::zip(operands, function.getBody().getArguments())) {
    mapping.map(std::get<0>(argument_pair), std::get<1>(argument_pair));
  }
  rewriter.setInsertionPointToStart(&function.getBody().front());
  ImplicitLocOpBuilder nb(loc, rewriter);
  for (auto op : ops) { nb.clone(*op, mapping); }
  SmallVector<::mlir::Value, 4> mapped_results;
  for (auto result : results) { mapped_results.push_back(mapping.lookup(result)); }
  rewriter.create<func::ReturnOp>(loc, mapped_results);
  if (symbol_table.lookup(func_name)) {
    emitError(loc) << func_name << " should not be at symbol table of ModuleOp";
    return nullptr;
  }
  return function;
}

NamedAttrList GetJitOpAttributes(::mlir::PatternRewriter& rewriter, StringRef op_name,
                                 int32_t input_size, int32_t output_size, Operation* op) {
  NamedAttrList attributes;
  attributes.set(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr(),
                 OpTrait::IsOpConfCompatible<void>::getDeviceTag(op));
  attributes.set(OpTrait::IsOpConfCompatible<void>::getDeviceNameAttr(),
                 OpTrait::IsOpConfCompatible<void>::getDeviceName(op));
  if (auto hierarchy = OpTrait::IsOpConfCompatible<void>::getHierarchy(op)) {
    attributes.set(OpTrait::IsOpConfCompatible<void>::getHierarchyAttr(), hierarchy);
  }
  attributes.set(OpTrait::IsOpConfCompatible<void>::getOpNameAttr(),
                 rewriter.getStringAttr(op_name));
  if (auto scope_symbol_id = OpTrait::IsOpConfCompatible<void>::getScopeSymbolID(op)) {
    attributes.set(OpTrait::IsOpConfCompatible<void>::getScopeSymbolIDAttr(), scope_symbol_id);
  }
  return attributes;
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

bool HasZeroPadding(mlir::ArrayAttr padding) {
  for (auto val : padding.getValue()) {
    if (val.cast<IntegerAttr>().getValue().getSExtValue() != 0) return false;
  }
  return true;
}

bool IsPaddingCouldBeAssimilatedIntoConv(::mlir::ArrayAttr padding_before,
                                         ::mlir::ArrayAttr padding_after,
                                         ::mlir::StringAttr data_format) {
  if (padding_before.size() == 4 && padding_after.size() == 4) {
    if (padding_before.getValue().equals(padding_after.getValue())) {
      if (data_format.str() == "channels_first") {
        return padding_before.getValue()[0].cast<IntegerAttr>().getValue().getSExtValue() == 0
               && padding_before.getValue()[1].cast<IntegerAttr>().getValue().getSExtValue() == 0;
      }
      if (data_format.str() == "channels_last") {
        return padding_before.getValue()[0].cast<IntegerAttr>().getValue().getSExtValue() == 0
               && padding_before.getValue()[3].cast<IntegerAttr>().getValue().getSExtValue() == 0;
      }
    }
  }
  return false;
}

IntegerAttr getSI64IntegerAttr(::mlir::PatternRewriter& rewriter, int64_t value) {
  return IntegerAttr::get(rewriter.getIntegerType(64, /*isSigned=*/true),
                          APInt(64, value, /*isSigned=*/true));
}

::llvm::SmallVector<::mlir::Value, 4> CreateConv2dAndErasePad(::mlir::PatternRewriter& rewriter,
                                                              OpResult conv_result,
                                                              OpResult pad_result) {
  if (auto conv_op = llvm::dyn_cast<oneflow::Conv2DOp>(conv_result.getDefiningOp())) {
    if (auto pad_op = llvm::dyn_cast<oneflow::PadOp>(pad_result.getDefiningOp())) {
      NamedAttrList attributes = conv_op->getAttrs();
      SmallVector<Value, 4> operands;
      operands.push_back(pad_op.x());
      operands.push_back(conv_op.weight());
      if (conv_op.bias()) operands.push_back(conv_op.bias());
      if (conv_op.bias_multiplier()) operands.push_back(conv_op.bias_multiplier());
      llvm::SmallVector<int32_t> padding_before_array;
      if (conv_op.data_formatAttr().getValue().str() == "channels_first") {
        for (auto val : pad_op.padding_before().getValue().take_back(2)) {
          padding_before_array.push_back(val.cast<IntegerAttr>().getValue().getSExtValue());
        }
      } else {
        padding_before_array.push_back(
            pad_op.padding_before().getValue()[1].cast<IntegerAttr>().getValue().getSExtValue());
        padding_before_array.push_back(
            pad_op.padding_before().getValue()[2].cast<IntegerAttr>().getValue().getSExtValue());
      }
      attributes.set(conv_op.padding_beforeAttrName(),
                     getSI32ArrayAttr(rewriter, padding_before_array));
      auto res = rewriter
                     .create<oneflow::Conv2DOp>(conv_op->getLoc(), conv_op->getResultTypes(),
                                                operands, attributes)
                     ->getResults();
      // pad op is expected to be erased if it is not used
      return res;
    }
  }
  return {};
}

::llvm::SmallVector<::mlir::Value, 4> CreateFusedBiasAddMaskScale(::mlir::PatternRewriter& rewriter,
                                                                  OpResult dropout_result,
                                                                  OpResult bias_add_result,
                                                                  Operation* mask) {
  if (auto dropout_op = llvm::dyn_cast<oneflow::DropoutOp>(dropout_result.getDefiningOp())) {
    if (auto bias_add_op = llvm::dyn_cast<oneflow::BiasAddOp>(bias_add_result.getDefiningOp())) {
      SmallVector<Value, 4> operands;
      operands.push_back(bias_add_op.a());
      operands.push_back(bias_add_op.b());
      operands.push_back(mask->getResults()[0]);
      NamedAttrList fused_bias_add_dropout_attributes = dropout_op->getAttrs();
      fused_bias_add_dropout_attributes.append(llvm::StringRef("axis"), bias_add_op.axisAttr());
      fused_bias_add_dropout_attributes.append(llvm::StringRef("scale"), dropout_op.rateAttr());
      fused_bias_add_dropout_attributes.erase(dropout_op.rateAttrName());
      auto res = rewriter
                     .create<oneflow::FusedBiasAddMaskScaleOp>(
                         dropout_op->getLoc(), dropout_op->getResultTypes().front(), operands,
                         fused_bias_add_dropout_attributes)
                     ->getResults();
      // bias_add and dropout op is expected to be erased if it is not used
      return res;
    }
  }
  return {};
}

mlir::IntegerAttr GetDefaultSeed(::mlir::PatternRewriter& rewriter) {
  const auto gen = CHECK_JUST(::oneflow::one::DefaultAutoGenerator());
  return getSI64IntegerAttr(rewriter, (int64_t)gen->current_seed());
}

LogicalResult InitTransposeAttributes(Operation* op, NamedAttrList& transpose_attributes,
                                      PatternRewriter& rewriter) {
  if (op->hasTrait<OpTrait::IsOpConfCompatible>()) {
    return OpTrait::IsOpConfCompatible<void>::saveToNamedAttrList(op, transpose_attributes);
  } else {
    op->emitError("must be a op of trait IsOpConfCompatible!");
    return failure();
  }
}

bool IsAddToOutputNone(ValueRange value) { return (int)value.size() > 0 ? false : true; }

llvm::SmallVector<int32_t> getChannelLastTransposePerm() { return {0, 2, 3, 1}; }

llvm::SmallVector<int32_t> getChannelFirstTransposePerm() { return {0, 3, 1, 2}; }

llvm::SmallVector<mlir::Value, 4> getInputOperandTransposeOp(NCHWCompatible op, Value val,
                                                             NamedAttrList transpose_attributes,
                                                             int num_transposed_operand,
                                                             PatternRewriter& rewriter) {
  std::string transpose_name = OpTrait::IsOpConfCompatible<void>::getOpName(op).str()
                               + "_transpose_input_" + std::to_string(num_transposed_operand);
  transpose_attributes.set(llvm::StringRef("op_name"), rewriter.getStringAttr(transpose_name));
  SmallVector<Value, 4> input_operands;
  input_operands.push_back(val);
  auto res = rewriter
                 .create<oneflow::TransposeOp>(op.getLoc(), val.getType(), input_operands,
                                               transpose_attributes)
                 ->getResults();
  return res;
}

TransposeOp getResultTransposeOp(NCHWCompatible op, Value val, NamedAttrList transpose_attributes,
                                 int num_transposed_result, PatternRewriter& rewriter) {
  std::string transpose_name = OpTrait::IsOpConfCompatible<void>::getOpName(op).str()
                               + "_transpose_output_" + std::to_string(num_transposed_result);
  transpose_attributes.set(llvm::StringRef("op_name"), rewriter.getStringAttr(transpose_name));
  SmallVector<Value, 4> operands;
  operands.push_back(val);
  TransposeOp transpose_op = rewriter.create<oneflow::TransposeOp>(op.getLoc(), val.getType(),
                                                                   operands, transpose_attributes);
  return transpose_op;
}

bool IsSameDtype(mlir::OpResult cast_result, mlir::Value input) {
  return cast_result.getType() == input.getType();
}

}  // namespace oneflow

}  // namespace mlir

#include "OneFlow/OneFlowPatterns.cpp.inc"

namespace mlir {

namespace oneflow {
struct AutoNhwcPattern : public OpInterfaceRewritePattern<NCHWCompatible> {
  explicit AutoNhwcPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern<NCHWCompatible>(context, /*benefit=*/1) {}

 public:
  LogicalResult matchAndRewrite(NCHWCompatible op, PatternRewriter& rewriter) const override {
    llvm::SmallVector<int32_t> perm = getChannelLastTransposePerm();
    llvm::SmallVector<int32_t> result_perm = getChannelFirstTransposePerm();

    NamedAttrList transpose_attributes;
    if (InitTransposeAttributes(op, transpose_attributes, rewriter).succeeded()) {
      transpose_attributes.append(llvm::StringRef("perm"), getSI32ArrayAttr(rewriter, perm));
    } else {
      return failure();
    }

    if (op.IsNCHW()) {
      // create transpose op for input operand
      SmallVector<Value, 4> tranposed_operands;
      llvm::DenseSet<Value> operand_transpose = op.OperandsToTranspose();
      int num_transposed_operand = 0;
      for (Value operand : op->getOperands()) {
        if (operand_transpose.find(operand) != operand_transpose.end()) {
          SmallVector<Value, 4> input_res = getInputOperandTransposeOp(
              op, operand, transpose_attributes, num_transposed_operand, rewriter);
          tranposed_operands.push_back(input_res[0]);
          num_transposed_operand += 1;
        }
      }
      // create NHWC op
      SmallVector<Value, 4> created_results = op.NchwToNhwc(tranposed_operands, rewriter);
      // create transpose op for results
      int num_transposed_result = 0;
      transpose_attributes.set(llvm::StringRef("perm"), getSI32ArrayAttr(rewriter, result_perm));
      llvm::DenseSet<Value> transpose_result = op.ResultsToTranspose();

      for (Value result : op->getOpResults()) {
        if (transpose_result.find(result) != transpose_result.end()) {
          if (auto result_transpose_op =
                  getResultTransposeOp(op, created_results[num_transposed_result],
                                       transpose_attributes, num_transposed_result, rewriter)) {
            result.replaceAllUsesWith(result_transpose_op);
            num_transposed_result += 1;
          } else {
            return failure();
          }
        }
      }
    }
    return success();
  }
};

void BroadcastMulOp::getCanonicalizationPatterns(RewritePatternSet& results, MLIRContext* context) {
  results.insert<BroadcastMulToScalarMulPattern>(context);
}

void AddLowerToLinalgMemRefPasses(PassManager& pm) {
  pm.addPass(createLowerOneFlowToTosaPass());                  // lower-oneflow-to-tosa
  pm.addPass(createCSEPass());                                 // cse
  pm.addNestedPass<func::FuncOp>(tosa::createTosaToLinalg());  // tosa-to-linalg-on-tensors
  pm.addNestedPass<func::FuncOp>(
      createLinalgElementwiseOpFusionPass());                       // linalg-fuse-elementwise-ops
  pm.addNestedPass<func::FuncOp>(createLinalgBufferizePass());      // linalg-bufferize
  pm.addNestedPass<func::FuncOp>(createTensorBufferizePass());      // tensor-bufferize
  pm.addPass(func::createFuncBufferizePass());                      // func-bufferize
  pm.addPass(bufferization::createBufferResultsToOutParamsPass());  // buffer-results-to-out-params
  pm.addPass(createCanonicalizerPass());                            // canonicalize
  pm.addNestedPass<func::FuncOp>(
      mlir::bufferization::createFinalizingBufferizePass());  // finalizing-bufferize
}

LogicalResult LowerModuleToLLVM(mlir::MLIRContext* context, ModuleOp module) {
  mlir::PassManager pm(context);
  AddLowerToLinalgMemRefPasses(pm);
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());  // convert-linalg-to-loops
  pm.addNestedPass<func::FuncOp>(createConvertSCFToCFPass());        // convert-scf-to-cf
  pm.addPass(createConvertLinalgToLLVMPass());                       // convert-linalg-to-llvm
  pm.addPass(createMemRefToLLVMPass());                              // convert-memref-to-llvm
  pm.addPass(createConvertFuncToLLVMPass());                         // convert-func-to-llvm
  pm.addPass(createReconcileUnrealizedCastsPass());                  // reconcile-unrealized-casts
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
  pm.addNestedPass<func::FuncOp>(
      createConvertLinalgToParallelLoopsPass());  // convert-linalg-to-parallel-loops
  pm.addPass(createMapSCFToGPUPass());            // gpu-greedy-parallel-loop-mapping
  pm.addPass(createParallelLoopToGpuPass());      // convert-parallel-loops-to-gpu
  pm.addPass(createGpuKernelOutliningPass());     // gpu-kernel-outlining
  pm.addNestedPass<func::FuncOp>(createBufferHostRegisterPass());  // buffer-host-register
  pm.addPass(createCanonicalizerPass());                           // canonicalize
  // -pass-pipeline='gpu.module([PASS1][PASS2]...)'
  pm.addNestedPass<gpu::GPUModuleOp>(createStripDebugInfoPass());        // strip-debuginfo
  pm.addNestedPass<gpu::GPUModuleOp>(createLowerAffinePass());           // lower-affine
  pm.addNestedPass<gpu::GPUModuleOp>(createLowerGpuOpsToNVVMOpsPass());  // convert-gpu-to-nvvm
  pm.addNestedPass<gpu::GPUModuleOp>(createSerializeToCubinPass());      // out-of-tree-gpu-to-cubin
  pm.addNestedPass<func::FuncOp>(createGpuCopyArgPass());                // buffer-host-register
  pm.addPass(createGpuToLLVMConversionPass());
  if (enable_ir_printing) pm.enableIRPrinting();
  return pm.run(module);
}

#endif  // WITH_MLIR_CUDA_CODEGEN

void populateFuserPasses(::mlir::RewritePatternSet& patterns) {
  patterns.add<MulCastPattern>(patterns.getContext());
}

void populateFuserForExistingOp(::mlir::RewritePatternSet& patterns) {
  patterns.add<FusedBiasAddGeluPattern>(patterns.getContext());
  patterns.add<FusedScaleTrilPattern>(patterns.getContext());
  patterns.add<FusedScaleTrilPattern2>(patterns.getContext());
  patterns.add<FusedPadConv2DPattern>(patterns.getContext());
  patterns.add<FusedBiasAddDropoutPattern>(patterns.getContext());
  patterns.add<NormalizationAddReluPattern>(patterns.getContext());
  patterns.add<DeleteSameDtypeCastOpPattern>(patterns.getContext());
  bool enable_nhwc = ::oneflow::ParseBooleanFromEnv("ONEFLOW_MLIR_PREFER_NHWC", false);
  if (enable_nhwc) { patterns.add<AutoNhwcPattern>(patterns.getContext()); }
}

void populateGpuHelperPatterns(::mlir::RewritePatternSet& patterns) {
  patterns.add<ReplaceCopyWithGPUPattern>(patterns.getContext());
}

}  // namespace oneflow

}  // namespace mlir
