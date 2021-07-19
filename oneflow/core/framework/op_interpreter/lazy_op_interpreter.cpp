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
#include "oneflow/core/framework/op_interpreter.h"

#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/op_arg_util.h"
#include "oneflow/core/framework/scope_util.h"
#include "oneflow/core/framework/session_util.h"
#include "oneflow/core/framework/symbol_storage_util.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_name_scope.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/eager/foreign_boxing_util.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/job/job_build_and_infer_ctx_mgr.h"

namespace oneflow {

namespace one {

std::string GetDeviceTagOfTensor(const std::shared_ptr<Tensor>& tensor) {
  if (tensor->is_cuda()) {
    return "gpu";
  } else {
    return "cpu";
  }
}

bool GetIsDynamicOfTensor(const std::shared_ptr<Tensor>& tensor) {
  if (tensor->is_consistent()) {
    return false;
  } else {
    return true;
  }
}

Maybe<void> GenParallelDistributionByTensor(ParallelDistribution* parallel_distribution,
                                            const std::shared_ptr<Tensor>& tensor) {
  parallel_distribution->clear_sbp_parallel();
  if (tensor->is_local()) {
    // NOTE(chengcheng):
    //   OneFlow Lazy is always consistent. LocalTensor is a special case of ConsistentTensor which
    //   placement is only this rank, and SbpParallel is Broadcast.
    parallel_distribution->add_sbp_parallel()->mutable_broadcast_parallel();
  } else {
    JUST(tensor->parallel_distribution())->ToProto(parallel_distribution);
  }
  return Maybe<void>::Ok();
}

Maybe<void> LazyInterpreter::ApplyImpl(const FeedInputOpExpr& op_expr, const TensorTuple& inputs,
                                       TensorTuple* outputs, const OpExprInterpContext& ctx) const {
  // NOTE(chengcheng): inputs[0] is the EagerTensor
  CHECK_EQ_OR_RETURN(inputs.size(), 1);
  CHECK_EQ_OR_RETURN(op_expr.input_size(), 1);
  const std::shared_ptr<Tensor>& input_tensor = inputs.at(0);
  CHECK_OR_RETURN(input_tensor->is_eager());

  const auto& scope = JUST(GetCurrentScope());
  int64_t scope_symbol_id = JUST(scope->symbol_id());

  OperatorConf op_conf;
  op_conf.set_name(op_expr.op_name());           // construct by python nn.Graph
  op_conf.set_scope_symbol_id(scope_symbol_id);  // TODO(chengcheng): NewScope by cur scope.
  op_conf.set_device_tag(GetDeviceTagOfTensor(input_tensor));
  // NOTE(chengcheng):
  //   We contruct InputOpConf instead of FeedInputOpConf because FeedInputOpExpr JUST for getting
  //   input EagerTensor.
  InputOpConf* input_conf = op_conf.mutable_input_conf();
  input_conf->set_out("out");
  InterfaceBlobConf* blob_conf = input_conf->mutable_blob_conf();

  input_tensor->shape()->ToProto(blob_conf->mutable_shape());
  blob_conf->set_data_type(input_tensor->dtype());
  blob_conf->set_is_dynamic(GetIsDynamicOfTensor(input_tensor));
  JUST(GenParallelDistributionByTensor(blob_conf->mutable_parallel_distribution(), input_tensor));

  auto infer_ctx = JUST(GetCurInferCtx());
  OpAttribute op_attr = *JUST(infer_ctx->AddAndInferConsistentOp(op_conf));

  const std::string& op_name = op_conf.name();

  // temp debug log
  std::cout << "cclog: Lazy nn.Graph AddOpName: " << op_name << std::endl
            << " and the origin op_conf is :" << op_conf.DebugString();

  int64_t parallel_desc_sym_id = JUST(scope->GetParallelDescSymbolId(op_conf));
  const std::shared_ptr<ParallelDesc>& blob_parallel_desc_sym =
      JUST(GetSymbol<cfg::ParallelConf, ParallelDesc>(parallel_desc_sym_id));

  // Check outputs num and setup output tensor properties.
  CHECK_EQ_OR_RETURN(outputs->size(), 1);
  CHECK_EQ_OR_RETURN(op_expr.output_size(), 1);

  const std::string obn = "out";  // NOTE(chengcheng): obn is NOT op_expr.indexed_obns
  const auto& parallel_attr =
      JUST(compatible_py::GetOpArgParallelAttribute(blob_parallel_desc_sym, op_attr, obn));
  const auto& blob_attr = JUST(compatible_py::GetOpArgBlobAttribute(op_attr, obn));

  CHECK_OR_RETURN(!outputs->at(0).get());
  (*outputs)[0] = JUST(OpInterpUtil::BuildTensor(blob_attr, parallel_attr, /*is_lazy=*/true));
  TensorNameScope::Global()->Record(outputs->at(0), op_name + "/" + obn);
  return Maybe<void>::Ok();
}

Maybe<void> LazyInterpreter::ApplyImpl(const FeedVariableOpExpr& op_expr, const TensorTuple& inputs,
                                       TensorTuple* outputs, const OpExprInterpContext& ctx) const {
  // NOTE(chengcheng): inputs[0] is the EagerTensor
  CHECK_EQ_OR_RETURN(inputs.size(), 1);
  CHECK_EQ_OR_RETURN(op_expr.input_size(), 1);
  const std::shared_ptr<Tensor>& input_tensor = inputs.at(0);
  CHECK_OR_RETURN(input_tensor->is_eager());

  const auto& scope = JUST(GetCurrentScope());
  int64_t scope_symbol_id = JUST(scope->symbol_id());

  OperatorConf op_conf;
  op_conf.set_name(op_expr.op_name());           // construct by python nn.Graph
  op_conf.set_scope_symbol_id(scope_symbol_id);  // TODO(chengcheng): NewScope by cur scope.
  op_conf.set_device_tag(GetDeviceTagOfTensor(input_tensor));
  // NOTE(chengcheng):
  //   We contruct VariableOpConf instead of FeedVariableOpConf because FeedVariableOpExpr JUST
  //   for getting input EagerTensor.
  VariableOpConf* var_conf = op_conf.mutable_variable_conf();
  var_conf->set_out("out");
  input_tensor->shape()->ToProto(var_conf->mutable_shape());
  var_conf->set_data_type(input_tensor->dtype());
  // NOTE(chengcheng): VariableOpConf initializer_conf is useless because variable is inited
  //   by EagerTensor.
  var_conf->mutable_initializer()->mutable_empty_conf();
  // TODO(chengcheng): GenerateParallelDistributionString by tensor.
  if (!input_tensor->requires_grad()) { var_conf->set_trainable(false); }
  // TODO(chengcheng, xuxiaoyu): Set L1/L2 RegularizerConf by nn.Graph Optimizer

  auto infer_ctx = JUST(GetCurInferCtx());
  OpAttribute op_attr = *JUST(infer_ctx->AddAndInferConsistentOp(op_conf));

  const std::string& op_name = op_conf.name();

  // temp debug log
  std::cout << "cclog: Lazy nn.Graph AddOpName: " << op_name << std::endl
            << " and the origin op_conf is :" << op_conf.DebugString();

  int64_t parallel_desc_sym_id = JUST(scope->GetParallelDescSymbolId(op_conf));
  const std::shared_ptr<ParallelDesc>& blob_parallel_desc_sym =
      JUST(GetSymbol<cfg::ParallelConf, ParallelDesc>(parallel_desc_sym_id));

  // Check outputs num and setup output tensor properties.
  CHECK_EQ_OR_RETURN(outputs->size(), 1);
  CHECK_EQ_OR_RETURN(op_expr.output_size(), 1);

  const std::string obn = "out";  // NOTE(chengcheng): obn is NOT op_expr.indexed_obns
  const auto& parallel_attr =
      JUST(compatible_py::GetOpArgParallelAttribute(blob_parallel_desc_sym, op_attr, obn));
  const auto& blob_attr = JUST(compatible_py::GetOpArgBlobAttribute(op_attr, obn));

  CHECK_OR_RETURN(!outputs->at(0).get());
  (*outputs)[0] = JUST(OpInterpUtil::BuildTensor(blob_attr, parallel_attr, /*is_lazy=*/true));
  // NOTE(chengcheng): Record variable op output LazyTenosr
  TensorNameScope::Global()->Record(outputs->at(0), op_name + "/" + obn);
  // NOTE(chengcheng): Record EagerTensor as variable tensor name
  TensorNameScope::Global()->Record(input_tensor, op_name + "/" + obn);
  return Maybe<void>::Ok();
}

Maybe<void> LazyInterpreter::ApplyImpl(const FetchOutputOpExpr& op_expr, const TensorTuple& inputs,
                                       TensorTuple* outputs, const OpExprInterpContext& ctx) const {
  // NOTE(chengcheng): inputs[0] is the LazyTensor
  CHECK_EQ_OR_RETURN(inputs.size(), 1);
  CHECK_EQ_OR_RETURN(op_expr.input_size(), 1);
  const std::shared_ptr<Tensor>& input_tensor = inputs.at(0);
  CHECK_OR_RETURN(input_tensor->is_lazy());
  // NOTE(chengcheng): Lazy always consistent.
  CHECK_OR_RETURN(input_tensor->is_consistent());
  const std::string& input_lbn = TensorNameScope::Global()->Lookup(input_tensor);
  CHECK_OR_RETURN(!input_lbn.empty());  // lbn must exist.

  const auto& scope = JUST(GetCurrentScope());
  int64_t scope_symbol_id = JUST(scope->symbol_id());

  OperatorConf op_conf;
  op_conf.set_name(op_expr.op_name());           // construct by python nn.Graph
  op_conf.set_scope_symbol_id(scope_symbol_id);  // TODO(chengcheng): NewScope by cur scope.
  op_conf.set_device_tag(GetDeviceTagOfTensor(input_tensor));
  // NOTE(chengcheng):
  //   We contruct OutputOpConf instead of FetchOutputOpConf because FetchOutputOpExpr JUST
  //   for get nn.Graph output LazyTensor.
  OutputOpConf* output_conf = op_conf.mutable_output_conf();
  output_conf->set_in(input_lbn);
  output_conf->set_out("out");
  InterfaceBlobConf* blob_conf = output_conf->mutable_blob_conf();
  input_tensor->shape()->ToProto(blob_conf->mutable_shape());
  blob_conf->set_data_type(input_tensor->dtype());
  blob_conf->set_is_dynamic(GetIsDynamicOfTensor(input_tensor));
  JUST(GenParallelDistributionByTensor(blob_conf->mutable_parallel_distribution(), input_tensor));

  auto infer_ctx = JUST(GetCurInferCtx());
  OpAttribute op_attr = *JUST(infer_ctx->AddAndInferConsistentOp(op_conf));

  const std::string& op_name = op_conf.name();

  // temp debug log
  std::cout << "cclog: Lazy nn.Graph AddOpName: " << op_name << std::endl
            << " and the origin op_conf is :" << op_conf.DebugString();

  int64_t parallel_desc_sym_id = JUST(scope->GetParallelDescSymbolId(op_conf));
  const std::shared_ptr<ParallelDesc>& blob_parallel_desc_sym =
      JUST(GetSymbol<cfg::ParallelConf, ParallelDesc>(parallel_desc_sym_id));

  // Check outputs num and setup output tensor properties.
  CHECK_EQ_OR_RETURN(outputs->size(), 1);
  CHECK_EQ_OR_RETURN(op_expr.output_size(), 1);

  const std::string obn = "out";  // NOTE(chengcheng): obn is NOT op_expr.indexed_obns
  const auto& parallel_attr =
      JUST(compatible_py::GetOpArgParallelAttribute(blob_parallel_desc_sym, op_attr, obn));
  const auto& blob_attr = JUST(compatible_py::GetOpArgBlobAttribute(op_attr, obn));

  CHECK_OR_RETURN(!outputs->at(0).get());
  // TODO(chengcheng): Build EagerLocalTensor if parllel attr is this rank.
  (*outputs)[0] = JUST(OpInterpUtil::BuildTensor(blob_attr, parallel_attr, /*is_lazy=*/false));
  return Maybe<void>::Ok();
}

Maybe<void> LazyInterpreter::ApplyImpl(const UserOpExpr& op_expr, const TensorTuple& inputs,
                                       TensorTuple* outputs, const OpExprInterpContext& ctx) const {
  CHECK_EQ_OR_RETURN(inputs.size(), op_expr.input_size());
  auto op_conf = JUST(OpInterpUtil::GenBuiltinOpConf(op_expr, ctx.attrs));
  // TODO(chengcheng): Handle special UserOp such as:
  //     1. [Source UserOp] : OFRecordReader, CoinFlip
  //     2. [Change Placement/ParallelDesc UserOp] : to/to_consistent/parallel_cast
  //     3. [Multi-Inputs & Different ParallelDesc for each input UserOp] : like there are 2 inputs,
  //             one from CPU and the other from GPU.
  //     ..., etc.

  const auto& scope = JUST(GetCurrentScope());
  int64_t old_scope_symbol_id = JUST(scope->symbol_id());
  // TODO(chengcheng): New parallel desc scope from all inputs tensors.
  op_conf->set_scope_symbol_id(old_scope_symbol_id);

  // NOTE(chengcheng):
  //   Normal UserOp inputs size >= 1 for infer parallel_desc.
  //   if inputs size == 1, need handle in SourceUserOp impl.
  CHECK_GE_OR_RETURN(inputs.size(), 1);
  const std::string device_tag = GetDeviceTagOfTensor(inputs.at(0));
  op_conf->set_device_tag(device_tag);
  for (int i = 0; i < inputs.size(); ++i) {
    const auto& input_tensor = inputs.at(i);
    CHECK_OR_RETURN(device_tag == GetDeviceTagOfTensor(input_tensor));
    const std::string& ibn = op_expr.indexed_ibns().at(i);
    const std::string& lbn = TensorNameScope::Global()->Lookup(inputs[i]);
    if (lbn.empty()) {
      CHECK_OR_RETURN(input_tensor->is_eager());  // NOTE(chengcheng): lazy_tensor MUST has lbn.

      // TODO(chengcheng):
      //     this is free EagerTensor which NOT captured by nn.Graph (inputs/params).
      //     Need Create a VariableOpConf for this inputs tensor, and Record name for itself.
      UNIMPLEMENTED();
    }
    CHECK_OR_RETURN(!lbn.empty());  // NOTE(chengcheng): lbn must not empty now.
    ReplaceInputLbnInOpCustomizedConf(op_conf.get(), ibn, lbn);
  }

  auto infer_ctx = JUST(GetCurInferCtx());
  // NOTE(chengcheng): MUST reset unique op name before InferCtx::AddOp
  const std::string new_op_name = *JUST(infer_ctx->NewUniqueOpNameByFunctionalOpConf(*op_conf));

  // temp debug log
  std::cout << "cclog: Lazy nn.Graph AddOpName: " << new_op_name << std::endl
            << " and the origin op_conf is :" << op_conf->DebugString();

  op_conf->set_name(new_op_name);
  OpAttribute op_attr = *JUST(infer_ctx->AddAndInferConsistentOp(*op_conf));

  int64_t parallel_desc_sym_id = JUST(scope->GetParallelDescSymbolId(*op_conf));
  const std::shared_ptr<ParallelDesc>& blob_parallel_desc_sym =
      JUST(GetSymbol<cfg::ParallelConf, ParallelDesc>(parallel_desc_sym_id));

  // Check outputs num and setup output tensor properties.
  CHECK_EQ_OR_RETURN(outputs->size(), op_expr.output_size());
  for (int i = 0; i < op_expr.output_size(); ++i) {
    const std::string& obn = op_expr.indexed_obns().at(i);
    const auto& parallel_attr =
        JUST(compatible_py::GetOpArgParallelAttribute(blob_parallel_desc_sym, op_attr, obn));
    const auto& blob_attr = JUST(compatible_py::GetOpArgBlobAttribute(op_attr, obn));
    if (!(outputs->at(i).get())) {
      (*outputs)[i] = JUST(OpInterpUtil::BuildTensor(blob_attr, parallel_attr, /*is_lazy=*/true));
    } else {
      // TODO(chengcheng, hjchen2) Reset shape, dtype and so on for InplaceUserOp.
      UNIMPLEMENTED();
    }
    TensorNameScope::Global()->Record(outputs->at(i), op_expr.op_name() + "/" + obn);
  }
  return Maybe<void>::Ok();
}

Maybe<void> LazyInterpreter::ApplyImpl(const FunctionOpExpr& op_expr, const TensorTuple& inputs,
                                       TensorTuple* outputs, const OpExprInterpContext& ctx) const {
  // TODO(hjchen2)
  OF_UNIMPLEMENTED() << "The type " << op_expr.op_type_name()
                     << " has not been supported in LazyInterpreter::Apply.";
  return Maybe<void>::Ok();
}

}  // namespace one
}  // namespace oneflow
