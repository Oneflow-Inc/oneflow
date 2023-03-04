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
#include "oneflow/ir/oneflow-translate/include/OneFlow/MLIROneFlowTranslation.h"
#include "oneflow/core/kernel/cuda_graph_support.h"
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/framework/variable_tensor_mgr.h"
#include "oneflow/core/operator/variable_op.h"
#include "oneflow/core/framework/sbp_context.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/framework/random_generator.h"
#include "oneflow/core/framework/variable_tensor_mgr.h"
#include "oneflow/core/operator/variable_op.h"
#include "oneflow/core/framework/sbp_context.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowUtils.h"
#include "OneFlow/Passes.h"
#include "OneFlow/OneFlowUtils.h"
#include "OneFlow/OneFlowPatternUtils.h"
#include "OneFlow/OneFlowSupport.h"
#include "OneFlow/SBP/SBPImporter.h"
#include "OneFlow/SBP/SBPAttributes.h"
#include "OneFlow/OKL/OKLOps.h"
#include "OneFlow/OKL/OKLTypes.h"
#include "OneFlow/OKL/Kernel/RegContext.h"
#include "OneFlow/OKM/Conversion/Conversion.h"
#include "OneFlow/Transform/TransposeHelpers.h"
#include "OneFlow/Transform/OutlineAndFuse.h"
#include "OneFlow/OneFlowPDLLPatterns.h"
#include "OneFlow/OKL/passes.h"
#include "OneFlow/OKL/OKLAttributes.h"
#include "OneFlow/OKM/passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/Transforms/RequestCWrappers.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Support/LLVM.h"

#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/TosaToLinalg/TosaToLinalg.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <iostream>
#include <string>

#ifdef WITH_MLIR_CUDA_CODEGEN
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/GPUCommon/GPUCommonPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Conversion/SCFToGPU/SCFToGPUPass.h"

#endif  // WITH_MLIR_CUDA_CODEGEN

#ifdef WITH_CUDA
// enable with_cuda_graphs
#include "oneflow/core/ep/cuda/cuda_stream.h"
#endif  // WITH_CUDA

namespace mlir {
namespace oneflow {

LLVM::LLVMPointerType GetPtr(::mlir::PatternRewriter& rewriter) {
  return LLVM::LLVMPointerType::get(IntegerType::get(rewriter.getContext(), 8));
}

template<typename T>
LogicalResult DumpAssembly(::mlir::PatternRewriter& rewriter, T op, StringRef func_name) {
  // TODO: now we only need one JIT engine
  auto parent_func_op = op->template getParentOfType<oneflow::Job>();
  if (!parent_func_op) { return failure(); }
  auto parent_module_op = parent_func_op->template getParentOfType<ModuleOp>();
  if (!parent_module_op) { return failure(); }
  SymbolTable symbol_table(parent_module_op);
  std::string mlir;
  llvm::raw_string_ostream os_mlir(mlir);
  if (auto found = symbol_table.lookup(func_name)) {
    found->print(os_mlir);
  } else {
    parent_module_op->dump();
    return op.emitError("symbol of jit function not found: " + op.op_name());
  }
  op->setAttr("mlir_assembly", rewriter.getStringAttr(mlir));
  return success();
}

LLVM::LLVMFuncOp DeclareKernelLaunchCInterface(::mlir::PatternRewriter& rewriter,
                                               mlir::Location loc, ModuleOp* module,
                                               StringRef c_api_callee, Type llvm_ptr_type) {
  LLVM::LLVMFuncOp func;
  if (!(func = module->lookupSymbol<LLVM::LLVMFuncOp>(c_api_callee))) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module->getBody());
    auto void_type = LLVM::LLVMVoidType::get(rewriter.getContext());
    auto func_type = LLVM::LLVMFunctionType::get(void_type, {llvm_ptr_type, llvm_ptr_type}, false);
    func = rewriter.create<LLVM::LLVMFuncOp>(loc, c_api_callee, func_type, LLVM::Linkage::External);

    func->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(rewriter.getContext()));
  }
  return func;
}

LLVM::GlobalOp DeclareOrGetGlobalString(::mlir::PatternRewriter& rewriter, mlir::Location loc,
                                        ModuleOp* module, StringRef func_name) {
  LLVM::GlobalOp global;
  StringRef variable = rewriter.getStringAttr(func_name + "_var");
  if (!(global = module->lookupSymbol<LLVM::GlobalOp>(variable))) {
    OpBuilder::InsertionGuard insertGuard(rewriter);
    rewriter.setInsertionPointToStart(module->getBody());
    auto type =
        LLVM::LLVMArrayType::get(IntegerType::get(rewriter.getContext(), 8), func_name.size());
    global =
        rewriter.create<LLVM::GlobalOp>(loc, type, /*isConstant=*/true, LLVM::Linkage::Internal,
                                        variable, rewriter.getStringAttr(func_name),
                                        /*alignment=*/0);
  }
  return global;
}

template<typename Wrap>
ModuleOp GetModuleOpFromJobBodyOp(Operation* op) {
  auto parent_func_op = op->getParentOfType<Wrap>();
  if (!parent_func_op) { return nullptr; }
  return parent_func_op->template getParentOfType<ModuleOp>();
}

func::FuncOp InsertKernelOFFuncOp(::mlir::PatternRewriter& rewriter, Operation* op,
                                  const std::string& func_name) {
  auto loc = op->getLoc();
  auto module = GetModuleOpFromJobBodyOp<func::FuncOp>(op);
  if (!module) {
    emitError(loc) << "null ModuleOp " << *op;
    return nullptr;
  }

  BlockAndValueMapping mapping;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());

  auto func_type =
      rewriter.getFunctionType(TypeRange(op->getOperandTypes()), TypeRange(op->getResultTypes()));
  func::FuncOp func = rewriter.create<func::FuncOp>(loc, func_name, func_type);
  func->setAttr("compiled", rewriter.getStringAttr("true"));
  func.getBody().emplaceBlock();
  for (auto& arg : func_type.getInputs()) { func.getBody().addArguments(arg, loc); }
  for (auto argument_pair :
       llvm::zip(ValueRange(op->getOperands()), func.getBody().getArguments())) {
    mapping.map(std::get<0>(argument_pair), std::get<1>(argument_pair));
  }
  rewriter.setInsertionPointToStart(&func.getBody().front());
  ImplicitLocOpBuilder new_block(loc, rewriter);
  new_block.clone(*op, mapping);
  SmallVector<::mlir::Value, 4> mapped_results;
  for (auto result : ValueRange(op->getResults())) {
    mapped_results.push_back(mapping.lookup(result));
  }
  rewriter.create<func::ReturnOp>(loc, mapped_results);
  return func;
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

bool HasZeroPadding(mlir::ArrayAttr padding) {
  for (auto val : padding.getValue()) {
    if (val.cast<IntegerAttr>().getValue().getSExtValue() != 0) return false;
  }
  return true;
}

NamedAttrList GetUserOpCommonAttrs(MLIRContext* ctx, const std::string& op_name) {
  NamedAttrList attrs;
  attrs.set(OpTrait::IsOpConfCompatible<void>::getOpNameAttr(), StringAttr::get(ctx, op_name));
  attrs.set(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr(), StringAttr::get(ctx, "cpu"));
  attrs.set(OpTrait::IsOpConfCompatible<void>::getDeviceNameAttr(),
            ArrayAttr::get(ctx, llvm::to_vector<8>(llvm::map_range(ArrayRef<StringRef>({"@0:0"}),
                                                                   [&](StringRef v) -> Attribute {
                                                                     return StringAttr::get(ctx, v);
                                                                   }))));
  return attrs;
}

struct ReplaceVariablePattern : public ::mlir::RewritePattern {
  explicit ReplaceVariablePattern(::mlir::MLIRContext* context)
      : ::mlir::RewritePattern("oneflow.variable", 1, context, {"oneflow.variable_ir"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation* op0,
                                        ::mlir::PatternRewriter& rewriter) const override {
    auto op = ::llvm::dyn_cast<oneflow::VariableOp>(op0);
    if (!op) return failure();
    NamedAttrList attrs;
    if (op.op_name().str().find("FreeEagerTensor") != std::string::npos) { return failure(); }
    attrs.set(StringAttr::get(getContext(), "value"),
              support::TensorToDenseElementsAttr(
                  CHECK_JUST(::oneflow::Singleton<::oneflow::VariableTensorMgr>::Get()->Get(
                      op.op_name().str(), ::oneflow::DType::Float())),
                  rewriter.getContext()));
    attrs.set(op.op_nameAttrName(), op.op_nameAttr());
    attrs.set(op.data_typeAttrName(), op.data_typeAttr());
    attrs.set(op.device_tagAttrName(), op.device_tagAttr());
    attrs.set(op.device_nameAttrName(), op.device_nameAttr());
    attrs.set(op.scope_symbol_idAttrName(), op.scope_symbol_idAttr());
    attrs.set(op.hierarchyAttrName(), op.hierarchyAttr());
    auto name = FrozenVariableOp::nd_sbpAttrName(
        OperationName(FrozenVariableOp::getOperationName(), rewriter.getContext()));

    auto parallel_attr = op.parallelAttr();
    attrs.set(name, SBPTranslation::ConvertSBPToString(rewriter, parallel_attr));
    auto op_new = rewriter.create<oneflow::FrozenVariableOp>(op->getLoc(), op.output().getType(),
                                                             ValueRange(), attrs);
    rewriter.replaceOp(op0, op_new->getResults());
    return ::mlir::success();
  }
};

struct ReplaceVariableIrPattern : public ::mlir::RewritePattern {
  explicit ReplaceVariableIrPattern(::mlir::MLIRContext* context)
      : ::mlir::RewritePattern("oneflow.variable_ir", 1, context, {"oneflow.variable"}) {}
  ::mlir::LogicalResult matchAndRewrite(::mlir::Operation* op0,
                                        ::mlir::PatternRewriter& rewriter) const override {
    auto op = ::llvm::dyn_cast<oneflow::FrozenVariableOp>(op0);
    if (!op) return failure();
    NamedAttrList attrs;
    const auto tensor_attr = op.value();
    attrs.set(StringAttr::get(getContext(), "shape"),
              rewriter.getArrayAttr(llvm::to_vector<8>(llvm::map_range(
                  tensor_attr.getType().cast<mlir::RankedTensorType>().getShape(),
                  [&](int64_t v) -> Attribute {
                    return IntegerAttr::get(rewriter.getIntegerType(64, /*isSigned=*/true),
                                            APInt(64, v, /*isSigned=*/true));
                  }))));
    attrs.set(StringAttr::get(getContext(), "data_type"),
              oneflow::DataTypeAttr::get(getContext(), oneflow::DataType::DT_Float));
    auto output_lbns_attr = rewriter.getStrArrayAttr({op.op_name().str() + "/out"});
    attrs.set(OpTrait::IsImportCompatible<void>::getOutputLBNsAttr(), output_lbns_attr);
    attrs.set(op.op_nameAttrName(), op.op_nameAttr());
    attrs.set(op.data_typeAttrName(), op.data_typeAttr());
    attrs.set(op.device_tagAttrName(), op.device_tagAttr());
    attrs.set(op.device_nameAttrName(), op.device_nameAttr());
    attrs.set(op.scope_symbol_idAttrName(), op.scope_symbol_idAttr());
    attrs.set(op.hierarchyAttrName(), op.hierarchyAttr());
    auto name = VariableOp::parallelAttrName(
        OperationName(VariableOp::getOperationName(), rewriter.getContext()));

    auto nd_size = op.hierarchy()->size();
    ArrayAttr nd_sbp = op.nd_sbp();
    std::vector<std::string> nd_sbp_str;
    std::for_each(nd_sbp.begin(), nd_sbp.end(), [&](Attribute elem) {
      if (auto sbp_str_attr = elem.dyn_cast<StringAttr>()) {
        nd_sbp_str.push_back(sbp_str_attr.str());
      }
    });
    attrs.set(name, SBPTranslation::ConvertNdSbpToPsig(rewriter, nd_sbp_str, nd_size));
    auto op_new = rewriter.create<oneflow::VariableOp>(op->getLoc(), op.output().getType(),
                                                       ValueRange(), attrs);
    const std::string tensor_name = op.op_nameAttr().str();
    const auto data_type = support::FromMLIRAttrToOFDataType(op.data_typeAttr());
    if (failed(data_type)) {
      op0->emitError(::llvm::formatv("unsupported data type: {0}",
                                     ConvertToString(op.data_typeAttr().getValue())));
      return ::mlir::failure();
    }
    auto var_tensor = CHECK_JUST(
        ::oneflow::Singleton<::oneflow::VariableTensorMgr>::Get()->Get(op.op_name().str()));
    if (var_tensor) {
      support::DenseElementsAttrToTensor(tensor_attr, op.device_tagAttr(), op.device_nameAttr(),
                                         var_tensor);
    } else {
      CHECK_JUST(::oneflow::Singleton<::oneflow::VariableTensorMgr>::Get()->Set(
          tensor_name,  // tensor_name can't be replaced by op.op_nameAttr().str() directly when
                        // compiling with gcc and I has no idea why.
                        // But it works when compiling with clang.
                        // Maybe temporary objects would be released earlier when using gcc.
          support::DenseElementsAttrToTensor(tensor_attr, op.device_tagAttr(),
                                             op.device_nameAttr()),
          CHECK_JUST(::oneflow::DType::Get(data_type.getValue()))));
    }
    // replaceOp may deallocate `op0` (and also `op`), so we should not use `op` after this call.
    rewriter.replaceOp(op0, op_new->getResults());
    return ::mlir::success();
  }
};

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
  transpose_attributes.set(llvm::StringRef(OpTrait::IsOpConfCompatible<void>::getOpNameAttr()),
                           rewriter.getStringAttr(transpose_name));
  SmallVector<Value, 4> input_operands;
  input_operands.push_back(val);
  auto res = rewriter
                 .create<oneflow::TransposeOp>(op.getLoc(), getNHWCType(val.getType()),
                                               input_operands, transpose_attributes)
                 ->getResults();
  return res;
}

TransposeOp getResultTransposeOp(NCHWCompatible op, Value val, NamedAttrList transpose_attributes,
                                 int num_transposed_result, PatternRewriter& rewriter) {
  std::string transpose_name = OpTrait::IsOpConfCompatible<void>::getOpName(op).str()
                               + "_transpose_output_" + std::to_string(num_transposed_result);
  transpose_attributes.set(llvm::StringRef(OpTrait::IsOpConfCompatible<void>::getOpNameAttr()),
                           rewriter.getStringAttr(transpose_name));
  SmallVector<Value, 4> operands;
  operands.push_back(val);
  TransposeOp transpose_op = rewriter.create<oneflow::TransposeOp>(
      op.getLoc(), getNCHWType(val.getType()), operands, transpose_attributes);
  return transpose_op;
}

bool IsInsertTransposeOpBefore(NCHWCompatible op, PatternRewriter& rewriter) {
  bool insert_transpose_op_flag = false;
  for (mlir::Value operand : op->getOperands()) {
    TransposeOp transposeInputOp = operand.getDefiningOp<TransposeOp>();
    if (!transposeInputOp) continue;
    const auto perm = transposeInputOp.permAttr();
    if (perm.size() == 4 && perm[0] == rewriter.getSI32IntegerAttr(0)
        && perm[1] == rewriter.getSI32IntegerAttr(3) && perm[2] == rewriter.getSI32IntegerAttr(1)
        && perm[3] == rewriter.getSI32IntegerAttr(2)) {
      insert_transpose_op_flag = true;
      break;
    }
  }
  return insert_transpose_op_flag;
}

}  // namespace oneflow

}  // namespace mlir

#include "OneFlow/OneFlowPatterns.cpp.inc"

namespace mlir {

namespace oneflow {

template<typename Op>
struct FusedConsecutiveAddPattern : public OpRewritePattern<Op> {
  explicit FusedConsecutiveAddPattern(mlir::MLIRContext* context)
      : OpRewritePattern<Op>(context, /*benefit=*/1) {}

 public:
  LogicalResult matchAndRewrite(Op op, PatternRewriter& rewriter) const override;
};

template<typename Op>
LogicalResult TryFusedConsecutiveAdd(Op op, const SmallVector<mlir::Value, 4>& opOperands,
                                     PatternRewriter& rewriter) {
  for (mlir::Value operand : opOperands) {
    if (!operand.getDefiningOp<AddNOp>() && !operand.getDefiningOp<Add2Op>()) { continue; }
    // check if the operand has only one user
    LogicalResult checkResult = [&]() {
      for (const auto& use : operand.getUses()) {
        if (use.getOwner() != op) { return failure(); }
      }
      return success();
    }();
    if (failed(checkResult)) { continue; }

    SmallVector<mlir::Value, 4> operands;
    SmallVector<mlir::Value, 4> inputOpOperands;
    mlir::Value inputOpResult;
    if (AddNOp addInputOp = operand.getDefiningOp<AddNOp>()) {
      inputOpOperands = addInputOp.in();
      inputOpResult = addInputOp.out();
    } else if (Add2Op addInputOp = operand.getDefiningOp<Add2Op>()) {
      inputOpOperands = {addInputOp.in0(), addInputOp.in1()};
      inputOpResult = addInputOp.out();
    }
    for (mlir::Value operand : opOperands) {
      if (operand != inputOpResult) {
        operands.push_back(operand);
      } else {
        operands.insert(operands.end(), inputOpOperands.begin(), inputOpOperands.end());
      }
    }
    auto new_op =
        rewriter.create<AddNOp>(op->getLoc(), op->getResultTypes(), operands, op->getAttrs());
    rewriter.replaceOp(op, new_op.out());
    return success();
  }
  return failure();
}

template<>
LogicalResult FusedConsecutiveAddPattern<AddNOp>::matchAndRewrite(AddNOp op,
                                                                  PatternRewriter& rewriter) const {
  return TryFusedConsecutiveAdd<AddNOp>(op, op.in(), rewriter);
}

template<>
LogicalResult FusedConsecutiveAddPattern<Add2Op>::matchAndRewrite(Add2Op op,
                                                                  PatternRewriter& rewriter) const {
  return TryFusedConsecutiveAdd<Add2Op>(op, {op.in0(), op.in1()}, rewriter);
}

struct AutoNhwcPattern : public OpInterfaceRewritePattern<NCHWCompatible> {
  explicit AutoNhwcPattern(mlir::MLIRContext* context)
      : OpInterfaceRewritePattern<NCHWCompatible>(context, /*benefit=*/1) {}

 public:
  LogicalResult matchAndRewrite(NCHWCompatible op, PatternRewriter& rewriter) const override {
    if (op->hasTrait<OpTrait::IsOpConfCompatible>()) {
      for (mlir::Value operand : op.OperandsToTranspose()) {
        if (operand.getType().cast<mlir::RankedTensorType>().getShape().size() != 4) {
          return failure();
        }
      }
      const auto device_name = OpTrait::IsOpConfCompatible<void>::getDeviceTag(op)
                                   .cast<mlir::StringAttr>()
                                   .getValue()
                                   .str();
      if (device_name == "cpu") { return failure(); }
    }
    llvm::SmallVector<int32_t> perm = getChannelLastTransposePerm();
    llvm::SmallVector<int32_t> result_perm = getChannelFirstTransposePerm();

    NamedAttrList transpose_attributes;
    if (InitTransposeAttributes(op, transpose_attributes, rewriter).succeeded()) {
      transpose_attributes.append(llvm::StringRef("perm"), getSI32ArrayAttr(rewriter, perm));
    } else {
      return failure();
    }
    // when op op has no sense of data_format and pre op is transpose, we greedily insert transpose
    // into this op, seeking more opportunities to eliminate transpose pattern.
    const bool greedily_transpose_flag = !op.IsNCHW() && IsInsertTransposeOpBefore(op, rewriter);

    if (op.IsNCHW() || greedily_transpose_flag) {
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

bool IsRedundantTransposeMatch(ArrayAttr pre, ArrayAttr afe, mlir::PatternRewriter& rewriter) {
  const auto prePerm = pre.getValue().vec();
  const auto afePerm = afe.getValue().vec();
  if (prePerm.size() == 4 && afePerm.size() == 4) {
    // handle nchw->nhwc->nchw: (0, 2, 3, 1) -> (0, 3, 1, 2)
    if (prePerm[0] == afePerm[0] && prePerm[1] == afePerm[3] && prePerm[2] == afePerm[1]
        && prePerm[3] == afePerm[2] && prePerm[0] == rewriter.getSI32IntegerAttr(0)
        && prePerm[1] == rewriter.getSI32IntegerAttr(2)
        && prePerm[2] == rewriter.getSI32IntegerAttr(3)
        && prePerm[3] == rewriter.getSI32IntegerAttr(1))
      return true;
    // handle nhwc->nchw->nhwc: (0, 3, 1, 2) -> (0, 2, 3, 1)
    if (prePerm[0] == afePerm[0] && prePerm[1] == afePerm[2] && prePerm[2] == afePerm[3]
        && prePerm[3] == afePerm[1] && prePerm[0] == rewriter.getSI32IntegerAttr(0)
        && prePerm[1] == rewriter.getSI32IntegerAttr(3)
        && prePerm[2] == rewriter.getSI32IntegerAttr(1)
        && prePerm[3] == rewriter.getSI32IntegerAttr(2))
      return true;
  }
  return false;
}

struct AutoNhwcEliminateRedundantTransposePattern : public mlir::OpRewritePattern<TransposeOp> {
  explicit AutoNhwcEliminateRedundantTransposePattern(mlir::MLIRContext* context)
      : OpRewritePattern<TransposeOp>(context, /*benefit=*/1) {}
  mlir::LogicalResult matchAndRewrite(TransposeOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    mlir::Value transposeInput = op.getOperand();
    TransposeOp transposeInputOp = transposeInput.getDefiningOp<TransposeOp>();

    if (!transposeInputOp
        || !IsRedundantTransposeMatch(op.permAttr(), transposeInputOp.permAttr(), rewriter)) {
      return failure();
    }
    rewriter.replaceOp(op, {transposeInputOp.getOperand()});
    return success();
  }
};

struct LowerToOKLPattern : public mlir::OpRewritePattern<func::FuncOp> {
  static LogicalResult LowerToOKLOp(::mlir::PatternRewriter& rewriter, Operation* op,
                                    func::FuncOp okl_func, int index) {
    auto op_type_name = op->getAttr("op_name").dyn_cast<StringAttr>();
    auto raw_func = op->getParentOfType<func::FuncOp>();
    if (!op_type_name) { return failure(); }
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(&okl_func.getBody().back());

    auto loc = op->getLoc();

    auto wrap_kernel = rewriter.create<okl::WrapperKernelOp>(loc, index);
    wrap_kernel.body().emplaceBlock();
    rewriter.setInsertionPointToEnd(&wrap_kernel.body().back());

    BlockAndValueMapping mapping;

    // map launcher_ctx from wrap func to block
    mapping.map(raw_func.getArgument(0), okl_func.getArgument(0));

    ImplicitLocOpBuilder new_block(loc, rewriter);
    for (auto arg : op->getOperands()) {
      auto define_op = arg.getDefiningOp();
      if (define_op->getName().getStringRef() == okl::GetTensorFromArgOp::getOperationName()) {
        new_block.clone(*define_op, mapping);
      } else {
        auto find = false;
        for (auto use : arg.getUsers()) {
          if (use->getName().getStringRef() == okl::GetTensorAsRetOp::getOperationName()) {
            find = true;
            auto index = use->getAttr("index").cast<IntegerAttr>().getInt();
            auto source = rewriter.create<okl::GetTensorFromRetOp>(op->getLoc(), arg.getType(),
                                                                   okl_func.getArgument(0), index);
            mapping.map(arg, source->getResult(0));
            break;
          }
        }
        if (!find) { op->emitError("Fail to find operand source"); }
      }
    }
    new_block.clone(*op, mapping);
    for (auto ret : op->getResults()) {
      auto find = false;
      for (auto use : ret.getUsers()) {
        if (use->getName().getStringRef() == okl::GetTensorAsRetOp::getOperationName()) {
          find = true;
          new_block.clone(*use, mapping);
          break;
        }
      }
      if (!find) { op->emitError("Fail to find result source"); }
    }
    rewriter.create<okl::ReturnOp>(loc);

    return success();
  }

  explicit LowerToOKLPattern(mlir::MLIRContext* context)
      : OpRewritePattern<func::FuncOp>(context, /*benefit=*/0) {}
  mlir::LogicalResult matchAndRewrite(func::FuncOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (!module) { LOG(FATAL) << "Not found module"; }
    if (module.lookupSymbol(okl_func::OKL_FUNC)) { return success(); }

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(op);
    auto& block = op.getBody().front();
    auto loc = op->getLoc();

    auto func_type = rewriter.getFunctionType(
        {mlir::okl::LauncherContextType::get(rewriter.getContext())}, TypeRange{});
    auto okl_func = rewriter.create<func::FuncOp>(loc, okl_func::OKL_FUNC, func_type);
    okl_func.getBody().emplaceBlock();
    okl_func.getBody().addArguments(mlir::okl::LauncherContextType::get(rewriter.getContext()),
                                    loc);

    auto index = 0;
    for (auto& op : block) {
      if (!op.hasAttr("op_name")) {
        if (op.getDialect()->getNamespace() == "okl") { continue; }
        if (isa<func::ReturnOp>(op)) { break; }
        op.emitError("Failed to parse this op in kernel launch wrap func.");
      }
      if (failed(LowerToOKLOp(rewriter, &op, okl_func, index))) {
        index += 1;
        op.emitError("Failed to lowering OneFlow op to okl dialect.");
        return failure();
      }
      index += 1;
    }

    rewriter.setInsertionPointToEnd(&okl_func.getBody().back());
    rewriter.create<func::ReturnOp>(loc);
    rewriter.eraseOp(op);
    return success();
  }
};

// {func, ins, outs_mapping}
std::tuple<func::FuncOp, std::vector<Value>, std::vector<std::vector<int>>>
CreateWrapFuncAndReturnWithIns(mlir::Location loc, std::vector<Operation*>& wrap_ops,
                               mlir::PatternRewriter& rewriter, int& name_index) {
  auto getProto =
      [&]() -> std::tuple<std::vector<Value>, std::vector<Value>, std::vector<std::vector<int>>> {
    std::vector<Value> whole_ins, whole_outs, ins, outs;
    std::vector<std::vector<int>> outs_mapping;
    for (auto op : wrap_ops) {
      auto operands = op->getOperands();
      auto results = op->getResults();
      for (auto it = operands.begin(); it != operands.end(); ++it) { whole_ins.push_back(*it); }

      std::vector<int> map;
      auto add_res = [&](mlir::OpResult res) {
        map.push_back(outs.size());
        outs.push_back(res);
      };
      for (auto it = results.begin(); it != results.end(); ++it) {
        whole_outs.push_back(*it);
        for (auto user : (*it).getUsers()) {
          if (std::find(wrap_ops.begin(), wrap_ops.end(), user) == wrap_ops.end()) {
            add_res(*it);
            break;
          }
        }
      }
      outs_mapping.push_back(map);
    }

    for (auto in : whole_ins) {
      if (std::find(whole_outs.begin(), whole_outs.end(), in) == whole_outs.end()) {
        ins.push_back(in);
      }
    }
    return {ins, outs, outs_mapping};
  };

  auto [ins, outs, map] = getProto();
  auto func_type = rewriter.getFunctionType(TypeRange(ValueRange(ArrayRef<Value>(ins))),
                                            TypeRange(ValueRange(ArrayRef<Value>(outs))));
  auto func_name = okm::func_name::GRAPH_NAME + std::to_string(name_index++);
  auto module = GetModuleOpFromJobBodyOp<Job>(wrap_ops[0]);
  if (!module) { LOG(FATAL) << "Fail to find parent ModuleOp"; }
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(module.getBody());
  auto function = rewriter.create<func::FuncOp>(loc, func_name, func_type);
  function->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(rewriter.getContext()));
  function.getBody().emplaceBlock();
  for (auto arg : ins) { function.getBody().addArgument(arg.getType(), loc); }

  BlockAndValueMapping mapping;
  for (auto args_pair : llvm::zip(ins, function.getBody().getArguments())) {
    mapping.map(std::get<0>(args_pair), std::get<1>(args_pair));
  }
  rewriter.setInsertionPointToStart(&function.getBody().front());
  ImplicitLocOpBuilder new_block(loc, rewriter);
  for (auto op : wrap_ops) { new_block.clone(*op, mapping); }

  SmallVector<::mlir::Value, 4> mapped_results;
  for (auto result : outs) { mapped_results.push_back(mapping.lookup(result)); }
  rewriter.create<func::ReturnOp>(loc, mapped_results);
  return {function, ins, map};
};

KernelLaunchOp ConsumeOpsToFunc(std::vector<Operation*>& wrap_ops, mlir::PatternRewriter& rewriter,
                                int& name_index) {
  if (wrap_ops.size() < 2) {
    wrap_ops.clear();
    return nullptr;
  }
  auto loc = wrap_ops.front()->getLoc();
  OpBuilder::InsertionGuard guard(rewriter);

  auto [wrap_func, wrap_ins, map] =
      CreateWrapFuncAndReturnWithIns(loc, wrap_ops, rewriter, name_index);

  auto func_name = wrap_func.getSymNameAttr();
  std::vector<NamedAttribute> attrs;
  for (auto attr : wrap_ops[0]->getAttrs()) {
    auto attr_list = {"scope_symbol_id", "device_tag", "device_name"};
    if (std::find(attr_list.begin(), attr_list.end(), attr.getName()) != attr_list.end()) {
      attrs.push_back(attr);
    }
  }

  attrs.emplace_back(rewriter.getStringAttr("op_name"), func_name);

  rewriter.setInsertionPointAfter(wrap_ops.back());
  auto func = rewriter.create<KernelLaunchOp>(wrap_ops[0]->getLoc(), wrap_func,
                                              ArrayRef<NamedAttribute>(attrs), wrap_ins);

  if (failed(DumpAssembly(rewriter, func, func_name))) {
    LOG(FATAL) << "Fail to dumping asm to kernel launch op.";
  }
  for (auto it : llvm::zip(map, wrap_ops)) {
    auto op = std::get<1>(it);
    auto list = std::get<0>(it);
    if (!list.size()) {
      op->dropAllUses();
      rewriter.eraseOp(op);
      continue;
    }
    std::vector<Value> vals;
    for (auto idx : list) { vals.push_back(func->getResult(idx)); }
    if (op->getNumResults() == vals.size()) {
      rewriter.replaceOp(op, vals);
    } else {  // if op has multi results but only some of them used outside, we need tackle with
              // mapper manually.
      int idx = 0;
      auto results = op->getResults();
      for (auto it = results.begin(); it != results.end(); ++it) {
        for (auto user : (*it).getUsers()) {
          if (std::find(wrap_ops.begin(), wrap_ops.end(), user) == wrap_ops.end()) {
            (*it).replaceAllUsesWith(func->getResult(list[idx]));
            idx += 1;
            break;
          }
        }
      }
    }
  }
  wrap_ops.clear();
  return func;
}
struct ExtractKernelLaunchTensorPattern : public mlir::OpRewritePattern<func::FuncOp> {
  static func::FuncOp ExtractArgTensors(func::FuncOp op, mlir::PatternRewriter& rewriter) {
    auto launcher_ctx_type = okl::LauncherContextType::get(rewriter.getContext());
    auto return_types = op.getBody().front().back().getOperandTypes();
    auto func_type = rewriter.getFunctionType({launcher_ctx_type}, return_types);

    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(), func_type);
    auto& body = func.getBody();

    body.emplaceBlock();
    body.addArgument(launcher_ctx_type, op->getLoc());
    auto launcher_ctx = body.getArgument(0);

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&body.front());

    BlockAndValueMapping mapping;
    for (const auto& arg : llvm::enumerate(op.getBody().getArguments())) {
      auto tensor = rewriter.create<okl::GetTensorFromArgOp>(func->getLoc(), arg.value().getType(),
                                                             launcher_ctx, arg.index());
      mapping.map(arg.value(), tensor);
    }

    ImplicitLocOpBuilder new_block(func->getLoc(), rewriter);
    for (auto& op : op.getBody().front().getOperations()) { new_block.clone(op, mapping); }
    rewriter.eraseOp(op);
    return func;
  }

  static func::FuncOp ExtractRetTensors(func::FuncOp op, mlir::PatternRewriter& rewriter) {
    auto& block = op.getBody().front();
    auto launcher_ctx = op.getArgument(0);
    auto& return_op = block.back();

    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(&return_op);

    std::vector<Value> returns;
    for (const auto& ret_val : llvm::enumerate(return_op.getOperands())) {
      auto new_ret = rewriter.create<okl::GetTensorAsRetOp>(
          op->getLoc(), ret_val.value().getType(), launcher_ctx, ret_val.value(), ret_val.index());
      returns.push_back(new_ret);
    }

    rewriter.replaceOpWithNewOp<func::ReturnOp>(&return_op, ValueRange{returns});
    return op;
  }

  explicit ExtractKernelLaunchTensorPattern(mlir::MLIRContext* context)
      : OpRewritePattern<func::FuncOp>(context, /*benefit=*/0) {}
  mlir::LogicalResult matchAndRewrite(func::FuncOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    if (op.getBody().getNumArguments()) {
      // skip if already converted
      if (op.getBody().getArgument(0).getType().isa<okl::LauncherContextType>()) {
        return success();
      }
    }
    op = ExtractArgTensors(op, rewriter);
    op = ExtractRetTensors(op, rewriter);
    return success();
  }
};

struct TrimReturnAsVoidPattern : public mlir::OpRewritePattern<func::FuncOp> {
  explicit TrimReturnAsVoidPattern(mlir::MLIRContext* context)
      : OpRewritePattern<func::FuncOp>(context, /*benefit=*/0) {}
  mlir::LogicalResult matchAndRewrite(func::FuncOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    if (op.getBody().front().back().getNumOperands() == 0) { return success(); }
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(op);

    auto func_type = rewriter.getFunctionType(op.getFunctionType().getInputs(), TypeRange{});
    auto func = rewriter.create<func::FuncOp>(op.getLoc(), op.getName(), func_type);

    BlockAndValueMapping bvm;
    op.getRegion().cloneInto(&func.getRegion(), bvm);

    auto& old_ret = func.getBody().front().back();
    rewriter.setInsertionPoint(&old_ret);
    rewriter.replaceOpWithNewOp<func::ReturnOp>(&old_ret);
    rewriter.eraseOp(op);
    return success();
  }
};

struct KernelLaunchPattern : public mlir::OpRewritePattern<oneflow::Job> {
  explicit KernelLaunchPattern(mlir::MLIRContext* context, bool trim = false)
      : OpRewritePattern<oneflow::Job>(context, /*benefit=*/0) {}

  // if the pre-packed ops is continuous with the current op, this current op will be packed with
  // pre-packed ops together.
  virtual bool IsConsecutive(std::vector<Operation*>&, mlir::Operation*) const { return true; };

  virtual bool IsPackagable(mlir::Operation* op) const {
    return GetModuleOpFromJobBodyOp<Job>(&(*op)) && op->getAttr("op_name")
           && dyn_cast<UserOpCompatible>(op)
           && op->getName().getStringRef() != KernelLaunchOp::getOperationName();
  }

  mlir::LogicalResult matchAndRewrite(oneflow::Job op,
                                      mlir::PatternRewriter& rewriter) const override {
    auto& ops = op->getRegion(0).front();
    if (ops.empty()) { return success(); }

    int name_index = 0;
    std::vector<Operation*> current_wrap_ops;
    for (auto op_it = ops.begin(); op_it != ops.end(); ++op_it) {
      auto current_op = &(*op_it);
      if (!IsPackagable(current_op)) {
        ConsumeOpsToFunc(current_wrap_ops, rewriter, name_index);
        continue;
      }

      if (!IsConsecutive(current_wrap_ops, current_op)) {
        ConsumeOpsToFunc(current_wrap_ops, rewriter, name_index);
      }
      current_wrap_ops.push_back(current_op);
    }
    if (!current_wrap_ops.empty()) { ConsumeOpsToFunc(current_wrap_ops, rewriter, name_index); }
    return success();
  }
};

struct KernelLaunchSimplePattern : public KernelLaunchPattern {
  explicit KernelLaunchSimplePattern(mlir::MLIRContext* context) : KernelLaunchPattern(context) {}

  bool IsSameDevice(std::vector<Operation*>& ops, mlir::Operation* op) const {
    if (ops.empty()) { return true; }

    auto device_tag = op->getAttr("device_tag").dyn_cast_or_null<StringAttr>();
    auto device_name = op->getAttr("device_name").dyn_cast_or_null<ArrayAttr>();
    auto cmp_device_tag = ops.front()->getAttr("device_tag").dyn_cast_or_null<StringAttr>();
    auto cmp_device_name = ops.front()->getAttr("device_name").dyn_cast_or_null<ArrayAttr>();

    if (!device_tag || !device_name || !cmp_device_tag || !cmp_device_name) { return false; }

    auto same_device_tag = device_tag.str() == cmp_device_tag.str();
    auto same_device_name =
        std::equal(device_name.begin(), device_name.end(), cmp_device_name.begin(),
                   [](const Attribute a, const Attribute b) {
                     auto a_str = a.dyn_cast_or_null<StringAttr>();
                     auto b_str = b.dyn_cast_or_null<StringAttr>();
                     if (!a_str || !b_str) { return false; }
                     return a_str.str() == b_str.str();
                   });

    return same_device_tag && same_device_name;
  }

  bool IsConsecutive(std::vector<Operation*>& ops, mlir::Operation* op) const override {
    if (ops.empty()) { return true; }
    return IsSameDevice(ops, op);
  }
};

struct KernelLaunchWithCudaGraphPattern : public KernelLaunchSimplePattern {
  explicit KernelLaunchWithCudaGraphPattern(mlir::MLIRContext* context)
      : KernelLaunchSimplePattern(context) {}

  bool IsOpCudaGraphSupport(mlir::Operation* op) const {
    ::oneflow::okl::RegContext reg_ctx(op);
    auto* kernel = const_cast<::oneflow::user_op::OpKernel*>(reg_ctx.GetKernel());
    return dynamic_cast<::oneflow::user_op::CudaGraphSupport*>(kernel);
  }

  bool IsSameCudaGraphSupport(std::vector<Operation*>& ops, mlir::Operation* op) const {
    if (ops.empty()) { return true; }
    auto cuda_support = IsOpCudaGraphSupport(op);
    return cuda_support == IsOpCudaGraphSupport(ops.front());
  }

  bool IsConsecutive(std::vector<Operation*>& ops, mlir::Operation* op) const override {
    if (ops.empty()) { return true; }
    return IsSameDevice(ops, op) && IsSameCudaGraphSupport(ops, op);
  }
};

void AddLowerToLinalgMemRefPasses(PassManager& pm) {
  pm.addPass(createConvertToSignlessForTosaPass());  // convert-to-signless-for-tosa
  pm.addNestedPass<func::FuncOp>(LLVM::createRequestCWrappersPass());  // llvm-request-c-wrappers
  pm.addPass(createConvertToSignlessForTosaPass());  // convert-to-signless-for-tosa
  pm.addPass(createLowerOneFlowToTosaPass());        // lower-oneflow-to-tosa
  pm.addNestedPass<func::FuncOp>(
      tosa::createTosaMakeBroadcastablePass());                // tosa-make-broadcastable
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
  mlir::oneflow::CheckEnableIRPrinting(pm);
  AddLowerToLinalgMemRefPasses(pm);
  pm.addNestedPass<func::FuncOp>(
      createConvertLinalgToParallelLoopsPass());  // convert-linalg-to-parallel-loops
  pm.addNestedPass<func::FuncOp>(createGpuMapParallelLoopsPass());  // gpu-map-parallel-loops
  pm.addPass(createParallelLoopToGpuPass());                        // convert-parallel-loops-to-gpu
  pm.addPass(createGpuLauchSinkIndexComputationsPass());
  pm.addPass(createGpuKernelOutliningPass());                      // gpu-kernel-outlining
  pm.addNestedPass<func::FuncOp>(createBufferHostRegisterPass());  // buffer-host-register
  pm.addPass(createCanonicalizerPass());                           // canonicalize
  // -pass-pipeline='gpu.module([PASS1][PASS2]...)'
  pm.addNestedPass<gpu::GPUModuleOp>(createStripDebugInfoPass());        // strip-debuginfo
  pm.addNestedPass<gpu::GPUModuleOp>(createLowerAffinePass());           // lower-affine
  pm.addNestedPass<gpu::GPUModuleOp>(createLowerGpuOpsToNVVMOpsPass());  // convert-gpu-to-nvvm
  pm.addNestedPass<gpu::GPUModuleOp>(createSerializeToCubinPass());      // out-of-tree-gpu-to-cubin
  pm.addNestedPass<func::FuncOp>(createGpuCopyArgPass());                // buffer-host-register
  pm.addPass(createGpuToLLVMConversionPass());
  pm.addPass(createReconcileUnrealizedCastsPass());  // reconcile-unrealized-casts
  return pm.run(module);
}

#endif  // WITH_MLIR_CUDA_CODEGEN

void populateFuserPasses(::mlir::RewritePatternSet& patterns) {
  rewrites::populateRewrites(patterns);
  constraints::populateConstraints(patterns);
  populateElementwiseFusionPatterns(patterns);
}

void populateWrapOpsToKernelLaunchPatterns(::mlir::RewritePatternSet& patterns,
                                           const std::string& mode) {
  if (mode == wrap_mode::SIMPLE) {
    patterns.add<KernelLaunchSimplePattern>(patterns.getContext());
  } else if (mode == wrap_mode::CUDA_GRAPH) {
#ifdef WITH_CUDA_GRAPHS
    patterns.add<KernelLaunchWithCudaGraphPattern>(patterns.getContext());
#else
    patterns.add<KernelLaunchPattern>(patterns.getContext());
#endif
  } else {
    LOG(FATAL) << "Found an unsupported mode in wrap-ops-to-kernel-launch pass";
  }
}

void populateFuserForExistingOp(::mlir::RewritePatternSet& patterns) {
  populateForwardOpPatterns(patterns);
  rewrites::populateRewrites(patterns);
  constraints::populateConstraints(patterns);
  populateNormalizationOpPatterns(patterns);
  patterns.add<FusedConsecutiveAddPattern<Add2Op>>(patterns.getContext());
  patterns.add<FusedConsecutiveAddPattern<AddNOp>>(patterns.getContext());
}

void populateAutoNhwcPatterns(::mlir::RewritePatternSet& patterns) {
  bool enable_nhwc = ::oneflow::ParseBooleanFromEnv("ONEFLOW_MLIR_PREFER_NHWC", false);
  if (enable_nhwc) {
    patterns.add<AutoNhwcPattern>(patterns.getContext());
    patterns.add<AutoNhwcEliminateRedundantTransposePattern>(patterns.getContext());
  }
}

void populateGpuHelperPatterns(::mlir::RewritePatternSet& patterns) {
  patterns.add<ReplaceCopyWithGPUPattern>(patterns.getContext());
}

void populatePreConvertInferenceOp(::mlir::RewritePatternSet& patterns) {
  patterns.add<ReplaceVariablePattern>(patterns.getContext());
}

void populateConvertInferenceOp(::mlir::RewritePatternSet& patterns) {
  populateFuseConv2DBatchNormPattern(patterns);
}

void populatePostConvertInferenceOp(::mlir::RewritePatternSet& patterns) {
  patterns.add<ReplaceVariableIrPattern>(patterns.getContext());
}

}  // namespace oneflow

}  // namespace mlir
