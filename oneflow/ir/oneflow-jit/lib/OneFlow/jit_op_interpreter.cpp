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

std::shared_ptr<one::Tensor> JitInterpreter::DispatchFunc(
    FuncOp func_op, const std::vector<std::shared_ptr<one::Tensor>>& arg_tensors) {
  llvm::DenseMap<Value, std::shared_ptr<Tensor>> mapping;
  // TODO: handle the case if there are more than one function in the module.
  std::shared_ptr<one::Tensor> ret_tensor;
  for (auto& op : func_op.getBody().getOps()) {
    if (dyn_cast<UserOpCompatible>(op)) {
      if (auto expr = GetExpr(&op)) {
        TensorTuple inputs(op.getOperands().size());
        for (const auto& indexed_operand : llvm::enumerate(op.getOperands())) {
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
        auto outputs = CHECK_JUST(OpInterpUtil::Dispatch<TensorTuple>(*expr.getValue(), inputs));
        if (outputs->size() != op.getResults().size()) {
          LOG(FATAL) << "The number of outputs of the op "
                     << " is not equal to the number of results.";
        }
        for (auto output_pair : llvm::zip(*outputs, op.getResults())) {
          auto output_tensor = std::get<0>(output_pair);
          Value output_result = std::get<1>(output_pair);
          CHECK(mapping.insert({output_result, output_tensor}).second);
        }
      } else {
        LOG(FATAL) << "The op " << op.getName().getStringRef().str() << " has not been supported.";
      }
    } else if (auto return_op = llvm::dyn_cast<ReturnOp>(op)) {
      ret_tensor = mapping.lookup(return_op.getOperands().front());
      CHECK(ret_tensor) << "The return tensor is not found.";
    }
  }
  return ret_tensor;
}

void JitInterpreter::Interrupt() {
  UNIMPLEMENTED();
  // FuncOp func_op = GetImporter().FinalizeProcessFunction();
  // if (ParseBooleanFromEnv("ONEFLOW_MLIR_STDOUT", false)) { module_->print(llvm::outs()); }
  // MlirTraceEnd();
  // std::vector<std::shared_ptr<one::Tensor>> ret{};
  // for (const auto& kv : GetImporter().GetIntermediateTensorsMapping()) {
  // ret.push_back(kv.second); } DispatchFunc(func_op, importer_.GetJitForwardArgs());
}

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
  CHECK(succeeded(ConvertUserOpInputs(op, user_op_adaptor, user_conf)));
  CHECK(succeeded(ConvertUserOpOutputs(op, user_op_adaptor, user_conf)));
  CHECK(succeeded(GetImporter().ConvertUserOpAttributes(op, user_op_adaptor, op_conf)));
  std::vector<std::string> indexed_ibns{};
  std::vector<std::string> indexed_obns{};
  InsertLbnSegmentIntoVec<OpTrait::AttrSizedOperandSegments>(op, indexed_ibns);
  InsertLbnSegmentIntoVec<OpTrait::AttrSizedResultSegments>(op, indexed_obns);
  auto expr = CHECK_JUST(UserOpExpr::New(user_op_adaptor.op_name().getValue().str(),
                                         std::move(*user_conf), indexed_ibns, indexed_obns));
  cached_user_op_exprs_.insert({hash, expr});
  return expr;
}

FuncOp JitInterpreter::Trace(ir::JitImporter& importer, const std::string& func_name,
                             const std::vector<std::shared_ptr<one::Tensor>>& arg_tensors,
                             const std::function<void()>& forward_func) {
  current_importer_ = &importer;  // TODO: extract function
  LOG(ERROR) << "importer reset";
  FuncOp func_op = importer.StartProcessFunc(func_name, arg_tensors);
  *one::MutJitEnabled() = true;
  forward_func();
  *one::MutJitEnabled() = false;
  return func_op;
}

std::shared_ptr<JitInterpreter> JitInterpreter::Get() {
  static auto interpreter = std::make_shared<one::JitInterpreter>();
  return interpreter;
}

}  // namespace one

}  // namespace oneflow

#endif  // WITH_MLIR
