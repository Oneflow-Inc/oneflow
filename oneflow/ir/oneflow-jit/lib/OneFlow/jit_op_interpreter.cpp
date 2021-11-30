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
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#ifdef WITH_MLIR
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/multi_client_session_context.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/op_arg_util.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/framework/symbol_storage_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_name_scope.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/eager/foreign_boxing_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"
#include "oneflow/core/vm/vm_util.h"
#include "oneflow/ir/oneflow-jit/include/OneFlow/jit_op_interpreter.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"

namespace oneflow {

namespace one {

using namespace mlir;

Maybe<void> JitInterpreter::Apply(const OpExpr& op_expr, const TensorTuple& inputs,
                                  TensorTuple* outputs, const OpExprInterpContext& ctx) const {
#define APPLY_IF(op_type)                                              \
  if (const auto* op = dynamic_cast<const op_type##Expr*>(&op_expr)) { \
    return ApplyImpl(*op, inputs, outputs, ctx);                       \
  }
  APPLY_IF(UserOp);
#undef APPLY_IF

  OF_UNIMPLEMENTED() << "The type " << op_expr.op_type_name()
                     << " has not been supported in JitInterpreter::Apply.";
  return Maybe<void>::Ok();
}

std::string GetDeviceTag(const std::shared_ptr<Tensor>& tensor) {
  if (tensor->is_cuda()) {
    return "gpu";
  } else {
    return "cpu";
  }
}

Maybe<const ParallelDesc> GetParallelDesc(const std::shared_ptr<Tensor>& tensor) {
  if (tensor->is_local()) {
    const auto& device = JUST(tensor->device());
    const auto& placement = JUST(Placement4Device(device));
    return placement.shared_from_symbol();
  } else {
    return JUST(tensor->parallel_desc()).shared_from_symbol();
  }
}

template<template<typename T> class Trait>
void InsertLbnSegmentIntoVec(Operation* op, std::vector<std::string>& indexed_bns) {
  std::vector<std::string> lbn_segment_keys;
  std::vector<int32_t> lbn_segment_sizes;
  CHECK(GetFilteredSegmentKeyAndSizes<Trait>(op, lbn_segment_keys, lbn_segment_sizes).succeeded());
  for (const auto& bn_size_pair : llvm::zip(lbn_segment_keys, lbn_segment_sizes)) {
    auto bn = std::get<0>(bn_size_pair);
    auto length = std::get<1>(bn_size_pair);
    for (size_t i = 0; i < length; i++) {
      const auto indexed_bn = bn + "_" + std::to_string(i);
      indexed_bns.push_back(indexed_bn);
    }
  }
}

void JitInterpreter::DispatchModule(
    ModuleOp module, const std::string& func_name,
    const std::vector<std::shared_ptr<one::Tensor>>& arg_tensors,
    std::vector<std::shared_ptr<one::Tensor>> returned_lazy_tensors) {
  llvm::DenseMap<Value, std::shared_ptr<Tensor>> mapping;
  // TODO: handle the case if there are more than one function in the module.
  ReturnOp return_op;
  SymbolTable symbol_table(module);
  auto function = symbol_table.lookup(func_name);
  if (!function) {
    module->dump();
    LOG(FATAL) << "The function " << func_name << " is not found in the module.";
  }
  const bool was_interrupted =
      function
          ->walk([&](mlir::Operation* op) {
            if (llvm::dyn_cast<mlir::oneflow::UserOp>(op) || op->hasAttr("op_type_name")) {
              if (auto expr = GetExpr(op)) {
                TensorTuple inputs(op->getOperands().size());
                for (const auto& indexed_operand : llvm::enumerate(op->getOperands())) {
                  auto index = indexed_operand.index();
                  auto operand = indexed_operand.value();
                  if (auto arg = operand.dyn_cast<mlir::BlockArgument>()) {
                    inputs[index] = arg_tensors[arg.getArgNumber()];
                  } else {
                    auto found = mapping.find(operand);
                    if (found->first) {
                      inputs[index] = found->second;
                    } else {
                      operand.dump();
                      LOG(FATAL) << "tensor not found";
                    }
                  }
                }
                // TODO: release the tensor if:
                // 1. it is the last use of the operand
                // 2. the tensor shared ptr has zero ref,
                // 3. it is not returned by the return op
                auto outputs =
                    CHECK_JUST(OpInterpUtil::Dispatch<TensorTuple>(*expr.getValue(), inputs));
                if (outputs->size() != op->getResults().size()) {
                  op->dump();
                  LOG(FATAL) << "The number of outputs of the op "
                             << " is not equal to the number of results.";
                }
                for (auto output_pair : llvm::zip(*outputs, op->getResults())) {
                  auto output_tensor = std::get<0>(output_pair);
                  Value output_result = std::get<1>(output_pair);
                  CHECK(mapping.insert({output_result, output_tensor}).second);
                }
                return WalkResult::advance();
              } else {
                return WalkResult::interrupt();
              }
            } else if (auto return_op_ = llvm::dyn_cast<ReturnOp>(op)) {
              return_op = return_op_;
              return WalkResult::advance();
            } else {
              return WalkResult::advance();
            }
          })
          .wasInterrupted();
  CHECK(!was_interrupted) << "JIT dispatch failure";
  llvm::DenseSet<Tensor*> tensors_to_materialize{};
  for (const auto& tensor_ref : returned_lazy_tensors) {
    tensors_to_materialize.insert(tensor_ref.get());
  }
  for (const auto& indexed_return_value : llvm::enumerate(return_op->getOperands())) {
    auto value = indexed_return_value.value();
    auto found = mapping.find(value);
    CHECK(found != mapping.end()) << "tensor not found";
    auto tensor_i = returned_lazy_tensors[indexed_return_value.index()];
    if (auto tensor_ref = std::dynamic_pointer_cast<ir::TensorRef>(tensor_i)) {
      if (tensors_to_materialize.contains(tensor_ref.get())) {
        tensor_ref->ResetTensor(mapping[value]);
      }
    } else {
      LOG(FATAL) << "tensor is not a TensorRef";
    }
  }
}

void JitInterpreter::Interrupt() {
  CHECK(GetImporter().LowerToOneFlowKernel().succeeded());
  if (ParseBooleanFromEnv("ONEFLOW_MLIR_STDOUT", false)) { module_->print(llvm::outs()); }
  MlirTraceEnd();
  auto ret = GetImporter().GetReturnTensors();
  DispatchModule(*module_, GetJitFuncName(), GetJitForwardArgs(), {ret.begin(), ret.end()});
  GetImporter().ResetMappings();
}  // namespace one

Maybe<void> JitInterpreter::ApplyImpl(const UserOpExpr& op_expr, const TensorTuple& inputs,
                                      TensorTuple* outputs, const OpExprInterpContext& ctx) const {
  auto op_conf = JUST(OpInterpUtil::GenBuiltinOpConf(op_expr, ctx.attrs));
  const std::string device_tag = GetDeviceTag(inputs.at(0));
  const bool is_local = inputs.at(0)->is_local();
  const std::shared_ptr<const ParallelDesc> parallel_desc = JUST(GetParallelDesc(inputs.at(0)));
  op_conf->set_device_tag(device_tag);

  for (int i = 0; i < inputs.size(); ++i) {
    const auto& input_tensor = inputs.at(i);
    CHECK_OR_RETURN(device_tag == GetDeviceTag(input_tensor));
    CHECK_OR_RETURN(parallel_desc->EqualsIgnoringHierarchy(*JUST(GetParallelDesc(input_tensor))));
    CHECK_EQ_OR_RETURN(is_local, input_tensor->is_local());
  }
  CHECK_EQ_OR_RETURN(outputs->size(), op_expr.output_size());
  auto indexed_arg_name_and_index = op_expr.input_arg_tuple()->indexed_arg_name_and_index();
  CHECK_EQ_OR_RETURN(indexed_arg_name_and_index.size(), inputs.size());
  GetImporter().GetOrInsertFunc(GetJitFuncName());
  GetImporter().CreateOperandMapping(*op_conf, parallel_desc, op_expr.input_arg_tuple(), inputs,
                                     outputs);
  CHECK_OR_RETURN(GetImporter().ProcessUserOp(*op_conf).succeeded());
  return Maybe<void>::Ok();
}

llvm::Optional<std::shared_ptr<one::UserOpExpr>> JitInterpreter::GetExpr(Operation* op) {
  auto hash = OperationEquivalence::computeHash(
      op,
      /*hashOperands=*/OperationEquivalence::ignoreHashValue,
      /*hashResults=*/OperationEquivalence::ignoreHashValue, OperationEquivalence::IgnoreLocations);

  auto it = cached_user_op_exprs_.find(hash);
  if (it != cached_user_op_exprs_.end()) { return it->second; }
  mlir::oneflow::UserOpAdaptor user_op_adaptor(op->getOperands(), op->getAttrDictionary());
  const std::string op_name = user_op_adaptor.op_name().getValue().str();
  ::oneflow::OperatorConf op_conf;
  auto user_conf = op_conf.mutable_user_conf();
  if (succeeded(ConvertUserOpInputs(op, user_op_adaptor, user_conf))
      && succeeded(ConvertUserOpOutputs(op, user_op_adaptor, user_conf))
      && succeeded(GetImporter().ConvertUserOpAttributes(op, user_op_adaptor, op_conf))
      && succeeded(ConvertCtrlInputs(op, op_conf))) {
    std::vector<std::string> indexed_ibns{};
    std::vector<std::string> indexed_obns{};
    InsertLbnSegmentIntoVec<OpTrait::AttrSizedOperandSegments>(op, indexed_ibns);
    InsertLbnSegmentIntoVec<OpTrait::AttrSizedResultSegments>(op, indexed_obns);
    auto expr = CHECK_JUST(UserOpExpr::New(user_op_adaptor.op_name().getValue().str(),
                                           std::move(*user_conf), indexed_ibns, indexed_obns));
    cached_user_op_exprs_.insert({hash, expr});
    return expr;
  }
  return None;
}

void JitInterpreter::Trace(
    ir::JitImporter& importer, const std::string& func_name,
    const std::vector<std::shared_ptr<one::Tensor>>& arg_tensors,
    const std::function<std::vector<std::shared_ptr<one::Tensor>>(void)>& forward_func) {
  Start();
  LOG(ERROR) << "importer reset";
  current_importer_ = &importer;  // TODO: extract function
  JitFunctionContext jit_function_context_(func_name, arg_tensors);
  auto return_tensors = forward_func();
  CHECK(importer.LowerToOneFlowKernel().succeeded());
  MlirTraceEnd();
}

}  // namespace one

}  // namespace oneflow

#endif  // WITH_MLIR
