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
#include "OneFlow/jit_importer.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/Dialect/JIT/IR/OneFlowJITDialect.h"
#include "OneFlow/jit_op_interpreter.h"
#include "OneFlow/Passes.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/framework/user_op_def.h"

namespace mlir {

namespace oneflow {

namespace {

std::unique_ptr<::oneflow::BlobDesc> GetBlobDescFromMlirTensorType(TensorType tensor_type) {
  auto dtype = ::oneflow::kInvalidDataType;
  if (tensor_type.getElementType().isF32()) {
    dtype = ::oneflow::kFloat;
  } else {
    tensor_type.dump();
    LOG(FATAL) << "fail to get BlobDesc from TensorType";
  }
  auto shape_from_mlir =
      new ::oneflow::Shape({tensor_type.getShape().begin(), tensor_type.getShape().end()});
  return std::make_unique<::oneflow::BlobDesc>(*shape_from_mlir, dtype);
}

::oneflow::ParallelContext GetSingleDeviceParallelContext() {
  ::oneflow::ParallelContext parallel_ctx;
  parallel_ctx.set_parallel_id(0);
  parallel_ctx.set_parallel_num(1);
  return parallel_ctx;
}

template<template<typename T> class Trait>
void InsertLbnSegmentIntoMapping(Operation* op, ValueRange values,
                                 std::unordered_map<std::string, mlir::Value>& value_mapping_) {
  std::vector<std::string> lbn_segment_keys;
  std::vector<int32_t> lbn_segment_sizes;
  CHECK(GetFilteredSegmentKeyAndSizes<Trait>(op, lbn_segment_keys, lbn_segment_sizes).succeeded());
  auto operand_it = values.begin();
  for (const auto& bn_size_pair : llvm::zip(lbn_segment_keys, lbn_segment_sizes)) {
    auto bn = std::get<0>(bn_size_pair);
    auto length = std::get<1>(bn_size_pair);
    for (size_t i = 0; i < length; i++) {
      const auto indexed_bn = bn + "_" + std::to_string(i);
      CHECK(value_mapping_.emplace(indexed_bn, *operand_it).second) << "indexed_bn: " << indexed_bn;
      operand_it += 1;
    }
  }
}

}  // namespace

void ProcessFuncContext::InsertValueMapping(Tensor* t, mlir::Value v) {
  CHECK(value_mapping_.insert({t, v}).second);
}

mlir::Value ProcessFuncContext::GetValue(Tensor* t) { return value_mapping_.lookup(t); }

const llvm::DenseMap<Tensor*, mlir::Value>& ProcessFuncContext::GetValueMapping() {
  return value_mapping_;
}

void ProcessFuncContext::InsertIntermediateTensor(Value v, std::shared_ptr<TensorRef> r) {
  CHECK(intermediate_tensors_mapping_.insert({v, r}).second);
}

LogicalResult JitImporter::AppendDataInOperand(const std::string& key, const int32_t index,
                                               const std::string& lbn,
                                               std::vector<::mlir::Value>& operand_vec) {
  operand_vec.push_back(GetResultByBnAndIndex(key, index).getValue());
  return success();
}
LogicalResult JitImporter::AddDeviceName(const ::oneflow::OperatorConf& op,
                                         std::vector<NamedAttribute>& attr_vec) {
  const ::oneflow::ParallelConf& pc = GetProcessOpContext().GetParallelDesc()->parallel_conf();
  std::vector<llvm::StringRef> device_vec = {pc.device_name().begin(), pc.device_name().end()};
  attr_vec.push_back(
      GetBuilder().getNamedAttr("device_name", GetBuilder().getStrArrayAttr(device_vec)));
  if (pc.has_hierarchy()) {
    attr_vec.push_back(GetBuilder().getNamedAttr(
        "hierarchy",
        GetBuilder().getI64ArrayAttr({pc.hierarchy().dim().begin(), pc.hierarchy().dim().end()})));
  }
  return success();
}
Type JitImporter::GetTensorTypeOfLbn(const std::string& lbn) {
  ::oneflow::LogicalBlobId lbi = ::oneflow::GenLogicalBlobId(lbn);
  return GetProcessOpContext().GetResultTypeMapping().at(lbi.blob_name());
}

std::shared_ptr<Tensor> UnWrapTensorRefOrReturnSelf(const std::shared_ptr<Tensor>& tensor) {
  if (auto tensor_ref = std::dynamic_pointer_cast<TensorRef>(tensor)) {
    return tensor_ref->GetTensor();
  } else {
    return tensor;
  }
}

std::shared_ptr<Tensor> JitImporter::MakeIntermediateTensor(
    const std::string& lbn, Value result,
    const std::shared_ptr<const ParallelDesc>& parallel_desc) {
  auto tensor_type = result.getType().cast<TensorType>();
  auto dtype = ::oneflow::kInvalidDataType;
  if (tensor_type.getElementType().isF32()) {
    dtype = ::oneflow::kFloat;
  } else if (tensor_type.getElementType().isInteger(32)) {
    dtype = ::oneflow::kInt32;
  } else if (tensor_type.getElementType().isInteger(64)) {
    dtype = ::oneflow::kInt64;
  } else {
    GetModule().dump();
    result.dump();
    LOG(FATAL) << "fail to create tensor";
  }
  const auto& device = CHECK_JUST(::oneflow::Device::MakeDeviceByParallelDesc(*parallel_desc));
  auto shape_from_mlir =
      new ::oneflow::Shape({tensor_type.getShape().begin(), tensor_type.getShape().end()});
  auto shape = std::make_shared<::oneflow::Shape>();
  shape.reset(shape_from_mlir);
  auto lazy_tensor = CHECK_JUST(
      ::oneflow::one::MirroredTensor::MakeTensor(shape, dtype, device, /* is_lazy */ true,
                                                 /* requires_grad= */ false, /* is_leaf= */ true));
  // tensor_ref holds an lazy or eager tensor
  auto tensor_ref = std::make_shared<TensorRef>(lazy_tensor);
  SaveIntermediate(result, tensor_ref);
  TrackTensorAndValue(lazy_tensor.get(), result);
  TrackTensorAndValue(tensor_ref.get(), result);
  return tensor_ref;
}
LogicalResult JitImporter::InsertOpResults(const ::oneflow::OperatorConf& op_conf,
                                           Operation* created_op) {
  auto output_lbns = created_op->getAttrOfType<ArrayAttr>("output_lbns");
  CHECK_EQ(output_lbns.size(), process_op_context_.GetOutputs()->size());
  for (const auto& data_out : llvm::enumerate(mlir::oneflow::GetDataOutputResults(created_op))) {
    auto lbn = output_lbns[data_out.index()].dyn_cast<StringAttr>().getValue().str();
    auto tensor =
        MakeIntermediateTensor(lbn, data_out.value(), process_op_context_.GetParallelDesc());
    (*process_op_context_.GetOutputs())[data_out.index()] = tensor;
  }
  return success();
}

FuncOp JitImporter::StartProcessFunc(
    llvm::StringRef func_name, const std::vector<std::shared_ptr<::oneflow::one::Tensor>>& args) {
  process_func_context_ = ProcessFuncContext(func_name, args);
  auto result_types = llvm::SmallVector<Type, 8>();
  SymbolTable symbol_table(GetModule());
  FuncOp found_func = symbol_table.lookup<FuncOp>(func_name);
  CHECK(!found_func) << "func exists: " << func_name.str();
  auto arg_tensors = GetJitForwardArgs();
  auto arg_types = llvm::SmallVector<Type, 8>();
  for (const auto& arg_tensor : arg_tensors) {
    auto mlir_dtype = GetTypeFromOneFlowDataType(arg_tensor->dtype()->data_type());
    auto mlir_tensor_type =
        RankedTensorType::get(ArrayRef<int64_t>(arg_tensor->shape()->dim_vec().begin(),
                                                arg_tensor->shape()->dim_vec().end()),
                              mlir_dtype.getValue());
    arg_types.push_back(mlir_tensor_type);
  }
  auto func_type = GetBuilder().getFunctionType(arg_types, llvm::NoneType());
  FuncOp function = mlir::FuncOp::create(GetRootLocation(), func_name, func_type);
  auto entryBlock = function.addEntryBlock();
  CHECK_EQ(arg_tensors.size(), function.body().getArguments().size());
  for (auto argument_pair : llvm::zip(arg_tensors, function.body().getArguments())) {
    // TODO: don't check here, because one tensor could be passed as multiple arguments
    TrackTensorAndValue(std::get<0>(argument_pair).get(), std::get<1>(argument_pair));
  }
  GetBuilder().setInsertionPointToStart(entryBlock);
  GetModule().push_back(function);
  return function;
}

llvm::Optional<TensorType> JitImporter::GetMlirTensorTypeFromBlobDesc(const BlobDesc& blob_desc) {
  if (auto t = GetTypeFromOneFlowDataType(blob_desc.data_type())) {
    return RankedTensorType::get(
        ArrayRef<int64_t>(blob_desc.shape().dim_vec().begin(), blob_desc.shape().dim_vec().end()),
        t.getValue());
  } else {
    return llvm::None;
  }
}

void JitImporter::CreateOperandMapping(const ::oneflow::OperatorConf& op_conf,
                                       const std::shared_ptr<const ParallelDesc> parallel_desc,
                                       const std::shared_ptr<const ArgTuple>& input_arg_tuple,
                                       const TensorTuple& inputs, TensorTuple* outputs) {
  process_op_context_ = ProcessOpContext(input_arg_tuple, inputs, outputs, parallel_desc);
  ::oneflow::HashMap<std::string, std::unique_ptr<BlobDesc>> lbi2logical_blob_desc_;
  auto op = CHECK_JUST(ConstructOp(op_conf));
  for (auto pair : llvm::zip(input_arg_tuple->indexed_bns(),
                             input_arg_tuple->indexed_arg_name_and_index(), inputs)) {
    const auto& indexed_bn = std::get<0>(pair);
    const auto& indexed_arg_name_and_index = std::get<1>(pair);
    const auto& tensor = std::get<2>(pair);
    if (auto result = GetResultByBnAndIndex(indexed_arg_name_and_index.first,
                                            indexed_arg_name_and_index.second)) {
      CHECK(
          GetProcessOpContext().GetOperandMapping().emplace(indexed_bn, result.getValue()).second);
    } else {
      LOG(FATAL) << "result not found, indexed_bn: " << indexed_bn << ", tensor: " << tensor.get()
                 << ", shape: " << tensor->shape()->DebugStr()
                 << ", dtype: " << tensor->dtype()->name()
                 << ", op conf: " << op_conf.DebugString();
    }
  }
  // TODO: refine here
  auto GetLogicalBlobDesc4BnInOp = [&](const std::string& bn) -> BlobDesc* {
    if (lbi2logical_blob_desc_.find(bn) == lbi2logical_blob_desc_.end()) {
      auto operand_it = GetProcessOpContext().GetOperandMapping().find(bn);
      if (operand_it == GetProcessOpContext().GetOperandMapping().end()) {
        auto blob_desc = std::make_unique<BlobDesc>(::oneflow::kInvalidDataType);
        CHECK(lbi2logical_blob_desc_.emplace(bn, std::move(blob_desc)).second);
      } else {
        auto found = mlir::oneflow::GetBlobDescFromMlirTensorType(
            operand_it->second.getType().cast<TensorType>());
        CHECK(lbi2logical_blob_desc_.emplace(bn, std::move(found)).second);
      }
    }
    return lbi2logical_blob_desc_.at(bn).get();
  };
  CHECK_JUST(op->InferLogicalOutBlobDescs(GetLogicalBlobDesc4BnInOp, *parallel_desc));
  static ::oneflow::ParallelContext parallel_ctx = mlir::oneflow::GetSingleDeviceParallelContext();
  for (auto& kv : lbi2logical_blob_desc_) {
    CHECK(GetProcessOpContext()
              .GetResultTypeMapping()
              .emplace(kv.first, GetMlirTensorTypeFromBlobDesc(*kv.second).getValue())
              .second);
  }
}

llvm::Optional<mlir::Value> JitImporter::GetResultByBnAndIndex(const std::string& bn,
                                                               const int32_t index) {
  auto idx = GetProcessOpContext().GetInputArgTuple()->TensorTupleIndex4ArgNameAndIndex(bn, index);
  auto tensor = GetProcessOpContext().GetInputs()[idx];
  auto result_it = GetValueMapping().find(UnWrapTensorRefOrReturnSelf(tensor).get());
  if (result_it == GetValueMapping().end()) {
    return llvm::None;
  } else {
    return result_it->second;
  }
}

FuncOp JitImporter::FinalizeProcessFunction(
    std::shared_ptr<::oneflow::one::Tensor> returned_tensor) {
  llvm::SmallVector<Value, 8> return_values{};
  auto v = GetValueMapping().lookup(UnWrapTensorRefOrReturnSelf(returned_tensor).get());
  CHECK(v) << "returned tensor not found";
  return_values.push_back(v);
  SymbolTable symbol_table(GetModule());
  auto func_name = GetJitFuncName();
  auto function = symbol_table.lookup<FuncOp>(func_name);
  if (!function) { LOG(FATAL) << "function not found: " << func_name; }

  auto return_op = GetBuilder().create<ReturnOp>(GetModule()->getLoc(), return_values);
  CHECK(return_op);

  llvm::SmallVector<Type, 8> out_types{};
  for (auto v : return_values) { out_types.push_back(v.getType()); }
  auto new_func_type = GetBuilder().getFunctionType(function.getType().getInputs(), out_types);
  function.setType(new_func_type);

  llvm::hash_code function_hash{};
  auto it = func_hash_symbol_mapping_.find(function_hash);
  if (it != func_hash_symbol_mapping_.end()) {
    function.erase();
    llvm::errs() << "cache hit: " << func_name << " -> " << it->second << "\n";
    return it->second;
  }
  llvm::errs() << function.sym_name() << ", "
               << "hash: " << function_hash << "\n";

  mlir::PassManager pm(GetMLIRContext(), /*operationName=*/"builtin.func");
  pm.addPass(::mlir::createCanonicalizerPass());
  // pm.addNestedPass<mlir::FuncOp>(::mlir::oneflow::createReturnAllLeaveResultPass());
  // pm.addNestedPass<mlir::FuncOp>(::mlir::oneflow::createCreateComputeCtxPass());
  pm.addPass(::mlir::oneflow::createFuseIntoExistingOpPass());
  CHECK(pm.run(function).succeeded());

  func_hash_symbol_mapping_.insert({function_hash, function});
  return function;
}

}  // namespace oneflow

}  // namespace mlir

namespace oneflow {

namespace one {

namespace ir {

OwningOpRef<ModuleOp> CreateJitModule(MLIRContext* context) {
  context->loadDialect<mlir::oneflow::OneFlowDialect>();
  context->loadDialect<mlir::oneflow::jit::OneFlowJITDialect>();
  context->loadDialect<StandardOpsDialect>();
  context->loadDialect<LLVM::LLVMDialect>();
  OwningOpRef<ModuleOp> module(
      ModuleOp::create(FileLineColLoc::get(context, "", /*line=*/0, /*column=*/0)));
  return module;
}

}  // namespace ir

}  // namespace one

}  // namespace oneflow
