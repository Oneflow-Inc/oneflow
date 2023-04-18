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
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/nn_graph.h"
#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_interpreter.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/core/framework/local_tensor_infer_cache.h"
#include "oneflow/core/framework/global_tensor_infer_cache.h"
#include "oneflow/core/boxing/eager_boxing_interpreter_mgr.h"
#include "oneflow/core/framework/consistency_check.h"
#include "oneflow/core/framework/tensor_global_id.h"
#include "oneflow/core/framework/nd_sbp.h"
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/boxing/eager_boxing_logger.h"
#include "oneflow/core/framework/global_tensor_infer_cache.h"

namespace oneflow {
namespace one {

using Env = std::map<std::string, std::shared_ptr<Tensor>>;
using NameToParallelDescType = std::map<std::string, Symbol<ParallelDesc>>;

Maybe<Env> InitEnv(const one::TensorTuple& graph_inputs, const std::shared_ptr<NNGraph>& graph) {
  Env env;
  for (const auto& [name, tensor] : graph->variable_op_name2tensor()) {
    env.emplace(name + "/out", tensor);
  }
  for (size_t i = 0; i < graph->inputs_op_names().size(); ++i) {
    const auto& name = graph->inputs_op_names()[i];
    env.emplace(name + "/out", JUST(VectorAt(graph_inputs, i)));
  }
  return env;
}

Maybe<UserOpExpr> OpConfToUserOpExpr(const OperatorConf& op_conf) {
  CHECK_OR_RETURN(op_conf.has_user_conf());
  const auto& user_conf = op_conf.user_conf();
  auto builder = OpBuilder(user_conf.op_type_name());
  for (const auto& pair : user_conf.attr()) { builder.Attr(pair.first, pair.second); }
  for (const auto& pair : user_conf.input()) {
    // ignore "UserSourceOpTickInput"
    if (pair.first == "UserSourceOpTickInput") { continue; }
    builder.Input(pair.first, pair.second.s_size());
  }
  for (const auto& pair : user_conf.output()) { builder.Output(pair.first, pair.second.s_size()); }
  return JUST(builder.Build());
}

template<typename Func>
Maybe<std::pair<TensorTuple, OpArgsVector<std::string>>> GetInputTensors(
    const UserOpConf& user_conf, const Env& env, const Func& preprocess) {
  TensorTuple inputs;
  OpArgsVector<std::string> ibns;
  for (const auto& [ibn, ibs] : user_conf.input()) {
    if (ibn == "UserSourceOpTickInput") { continue; }
    const auto& tensor_names = ibs.s();
    for (int i = 0; i < tensor_names.size(); ++i) {
      inputs.emplace_back(preprocess(JUST(MapAt(env, tensor_names[i]))));
      ibns.emplace_back(ibn + '_' + std::to_string(i));
    }
  }
  return std::make_pair(inputs, ibns);
}

Maybe<std::pair<OpArgsVector<std::string>, OpArgsVector<std::string>>> GetOutputNamesOfOp(
    const UserOpConf& user_conf) {
  OpArgsVector<std::string> output_names;
  OpArgsVector<std::string> obns;
  for (const auto& [obn, obs] : user_conf.output()) {
    const auto& tensor_names = obs.s();
    for (int i = 0; i < tensor_names.size(); ++i) {
      output_names.emplace_back(tensor_names[i]);
      obns.emplace_back(obn + '_' + std::to_string(i));
    }
  }
  return std::make_pair(output_names, obns);
}

// Only support a limited subset of view ops for now
bool IsViewOp(const std::shared_ptr<UserOpExpr>& op) {
  return op->op_type_name() == "reshape" || op->op_type_name() == "expand_dims";
}

std::string ErrorString4Inputs(const TensorTuple& inputs,
                               const std::shared_ptr<UserOpExpr>& op_expr) {
  std::stringstream error_str;
  error_str << "Got input tensors with inconsistent attributes!\n"
            << "op_type_name: " << op_expr->op_type_name() << "\n"
            << "attributes of inputs is:\n";
  int32_t idx = 0;
  for (const auto& tensor : inputs) {
    if (tensor->is_local()) {
      error_str << "local";
    } else {
      error_str << "global";
    }
    if (++idx != inputs.size()) { error_str << ", "; }
  }
  return error_str.str();
}

Maybe<bool> GetEagerInterpreterType(const TensorTuple& inputs,
                                    const std::shared_ptr<UserOpExpr>& op_expr) {
  bool is_local = true;
  if (inputs.empty()) {
    CHECK_OR_RETURN(0) << "input empty";
  } else {
    if (inputs[0]->is_global()) {
      if (inputs.size() == 1) {
        // do nothing
      } else if (inputs.size() == 2) {
        CHECK_OR_RETURN(inputs[1]->is_global())      // NOLINT
            << ErrorString4Inputs(inputs, op_expr);  // unroll loop for efficiency
      } else if (inputs.size() == 3) {
        CHECK_OR_RETURN(inputs[1]->is_global())
            << ErrorString4Inputs(inputs, op_expr);  // unroll loop for efficiency
        CHECK_OR_RETURN(inputs[2]->is_global())
            << ErrorString4Inputs(inputs, op_expr);  // unroll loop for efficiency
      } else {
        for (const auto& tensor : inputs) {
          CHECK_OR_RETURN(tensor->is_global()) << ErrorString4Inputs(inputs, op_expr);
        }
      }
      is_local = false;
    } else {
      if (inputs.size() == 1) {
        // do nothing
      } else if (inputs.size() == 2) {
        CHECK_OR_RETURN(inputs.at(1)->is_local())
            << ErrorString4Inputs(inputs, op_expr);  // unroll loop for efficiency
      } else if (inputs.size() == 3) {
        CHECK_OR_RETURN(inputs.at(1)->is_local())
            << ErrorString4Inputs(inputs, op_expr);  // unroll loop for efficiency
        CHECK_OR_RETURN(inputs.at(2)->is_local())
            << ErrorString4Inputs(inputs, op_expr);  // unroll loop for efficiency
      } else {
        for (const auto& tensor : inputs) {
          CHECK_OR_RETURN(tensor->is_local()) << ErrorString4Inputs(inputs, op_expr);
        }
      }
    }
  }
  return is_local;
}

Maybe<void> RunViewOp(const std::shared_ptr<UserOpExpr>& op, Env& env, const TensorTuple& inputs,
                      const OpArgsVector<std::string>& output_names) {
  // eliminate the memcpy of view ops
  CHECK_OR_RETURN(IsViewOp(op));
  const std::shared_ptr<const LocalTensorInferResult> result =
      JUST([&]() -> Maybe<const LocalTensorInferResult> {
        LocalTensorMetaInferArgs infer_args;
        JUST(infer_args.Init(op->base_attrs(), JUST(inputs[0]->device()), inputs));
        return JUST(op->mut_local_tensor_infer_cache()->GetOrInfer(infer_args));
      }());
  const auto& output_shape = result->output_tensor_metas()[0]->shape();
  const auto output =
      JUST(view::BasicView(inputs[0], output_shape, JUST(inputs[0]->storage_offset())));
  env.emplace(output_names[0], output);
  return Maybe<void>::Ok();
}

namespace {

Maybe<bool> IsAllZeroSizeTensorMeta(const std::vector<Symbol<GlobalTensorMeta>>& tensor_metas) {
  if (tensor_metas.empty()) { return false; }
  for (const auto& tensor_meta : tensor_metas) {
    if (tensor_meta->shape().elem_cnt() != 0) { return false; }
  }
  return true;
}

constexpr auto* CachedIsAllZeroSizeTensorMeta =
    DECORATE(&IsAllZeroSizeTensorMeta, ThreadLocalCopiable);

class UserOpExprDeviceAndStreamInferContext final : public user_op::DeviceAndStreamInferContext {
 public:
  UserOpExprDeviceAndStreamInferContext(const UserOpExpr* user_op_expr,
                                        const GlobalTensorMetaInferArgs* infer_args,
                                        const Symbol<ParallelDesc>& op_parallel_desc)
      : user_op_expr_(user_op_expr),
        composed_attrs_(infer_args->attrs(), user_op_expr->base_attrs()),
        in_tensor_devices_(user_op_expr_->input_size()),
        out_tensor_devices_(user_op_expr_->output_size()) {
    for (int i = 0; i < user_op_expr_->input_size(); ++i) {
      const auto& parallel_desc =
          infer_args->input_global_tensor_metas().at(i).tensor_meta()->parallel_desc();
      in_tensor_devices_.at(i) = CHECK_JUST(GetTensorDevice(parallel_desc));
      out_tensor_devices_.at(i) = CHECK_JUST(GetTensorDevice(op_parallel_desc));
    }
  }

  const std::vector<std::pair<std::string, int32_t>>& inputs() const override {
    return user_op_expr_->indexed_input_pairs();
  }

  const std::vector<std::pair<std::string, int32_t>>& outputs() const override {
    return user_op_expr_->indexed_output_pairs();
  }

  Symbol<Device>* OutputTensorDevice4ArgNameAndIndex(const std::string& name,
                                                     int64_t index) override {
    const auto& arg_tuple = *user_op_expr_->output_arg_tuple();
    int32_t tuple_index = arg_tuple.TensorTupleIndex4ArgNameAndIndex(name, index);
    CHECK_GE(tuple_index, 0);
    CHECK_LT(tuple_index, user_op_expr_->output_size());
    return &out_tensor_devices_.at(tuple_index);
  }

  Symbol<Device> InputTensorDevice4ArgNameAndIndex(const std::string& name,
                                                   int64_t index) const override {
    const auto& arg_tuple = *user_op_expr_->input_arg_tuple();
    int32_t tuple_index = arg_tuple.TensorTupleIndex4ArgNameAndIndex(name, index);
    CHECK_GE(tuple_index, 0);
    CHECK_LT(tuple_index, user_op_expr_->input_size());
    return in_tensor_devices_.at(tuple_index);
  }

 private:
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return composed_attrs_.Attr4Name(attr_name);
  }
  const UserOpExpr* user_op_expr_;
  const ComposedAttrMap composed_attrs_;
  std::vector<Symbol<Device>> in_tensor_devices_;
  std::vector<Symbol<Device>> out_tensor_devices_;
};

/* static */ Maybe<Symbol<Stream>> InferDeviceAndStream(
    const UserOpExpr& user_op_expr, const GlobalTensorMetaInferArgs& infer_args,
    const Symbol<ParallelDesc>& op_parallel_desc) {
  if (!user_op_expr.device_and_stream_infer_fn()) {
    Symbol<ParallelDesc> parallel_desc =
        infer_args.input_global_tensor_metas()[0].tensor_meta()->parallel_desc();
    return GetDefaultStreamByPlacement(parallel_desc);
  } else {
    UserOpExprDeviceAndStreamInferContext device_and_stream_ctx(&user_op_expr, &infer_args,
                                                                op_parallel_desc);
    return TRY(user_op_expr.device_and_stream_infer_fn()(&device_and_stream_ctx));
  }
}

Maybe<void> RawRunGlobalNormalOp(
    const std::shared_ptr<UserOpExpr>& op, const TensorTuple& inputs, TensorTuple* outputs,
    std::map<std::string, std::shared_ptr<Tensor>>& env, const OperatorConf& op_conf,
    const UserOpConf& user_conf, const OpArgsVector<std::string>& ibns,
    const OpArgsVector<std::string>& obns, const OpArgsVector<std::string>& output_names,
    const NdSbpSignature& ndsbp_signature, const Symbol<ParallelDesc>& op_parallel_desc) {
  CHECK_EQ_OR_RETURN(outputs->size(), op->output_size());
  static AttrMap empty_attr_map;
  const OpExprInterpContext ctx(empty_attr_map);
  CHECK_OR_RETURN(!inputs.empty());
  const auto& parallel_desc = JUST(inputs.at(0)->parallel_desc());

  Optional<int64_t> parallel_id;
  const auto& tensor_device = JUST(GetTensorDevice4CurrentProcessCtx(parallel_desc, &parallel_id));
  const auto& infer_args_ptr = JUST(GlobalTensorMetaInferArgs::New(ctx.attrs, inputs));
  const auto& infer_args = *infer_args_ptr;
  auto mut_result = std::make_unique<GlobalTensorInferResult>(op->input_size(), op->output_size());
  CHECK_EQ_OR_RETURN(outputs->size(), obns.size());
  CHECK_EQ_OR_RETURN(outputs->size(), op->output_size());
  CHECK_GT_OR_RETURN(infer_args.input_global_tensor_metas().size(), 0);
  for (int i = 0; i < outputs->size(); ++i) {
    if ((*outputs)[i]) {
      const auto& nd_sbp = JUST((*outputs)[i]->nd_sbp());
      JUST((*outputs)[i]->set_consumer_nd_sbp_constraint(nd_sbp));
    }
  }
  std::vector<OpArgMutGlobalTensorMeta> output_mut_metas(op->output_size());
  {
    // Infer OpArgMutGlobalTensorMeta.
    const auto& input_metas = infer_args.input_global_tensor_metas();
    JUST(op->InferLogicalTensorDesc(
        infer_args.attrs(), parallel_desc,
        [&](int32_t i) { return &*input_metas.at(i).tensor_meta(); },
        [&](int32_t i) { return output_mut_metas.at(i).mut_tensor_meta(); }));
  }
  const auto& output_metas = mut_result->mut_output_tensor_metas();
  CHECK_EQ_OR_RETURN(outputs->size(), output_metas->size());
  for (int i = 0; i < outputs->size(); ++i) {
    if (!outputs->at(i)) {
      const auto& output_mut_meta = JUST(VectorAt(output_mut_metas, i));
      const auto& shape = output_mut_meta.tensor_meta().shape();
      DataType data_type = output_mut_meta.tensor_meta().data_type();
      std::string lbn = JUST(VectorAt(obns, i));
      const auto& nd_sbp = SymbolOf(JUST(MapAt(ndsbp_signature.bn_in_op2nd_sbp(), lbn)));
      GlobalTensorMeta output_meta(shape, data_type, nd_sbp, op_parallel_desc);
      const auto& tensor_impl = JUST(EagerGlobalTensorImpl::New(
          SymbolOf(output_meta), tensor_device, parallel_id, false, false));
      (*outputs)[i].reset(new GlobalTensor(tensor_impl));

      GlobalTensorMeta tensor_meta(shape, data_type, nd_sbp, parallel_desc);
      output_metas->at(i) = SymbolOf(tensor_meta);
    } else {
      JUST((*outputs)[i]->set_consumer_nd_sbp_constraint(NullOpt));
    }
  }
  mut_result->set_stream(JUST(InferDeviceAndStream(*op, infer_args, op_parallel_desc)));

  vm::EagerBlobObjectList input_eager_blob_objects;
  TensorTuple boxing_outputs;
  const auto* mgr = Singleton<EagerBoxingInterpreterManager>::Get();
  auto* input_metas = mut_result->mut_input_tensor_metas();
  CHECK_EQ_OR_RETURN(inputs.size(), ibns.size());
  for (int i = 0; i < inputs.size(); ++i) {
    std::shared_ptr<Tensor> input_tensor = inputs[i];
    std::string lbn = JUST(VectorAt(ibns, i));
    const auto& logical_shape = input_tensor->shape();
    CHECK_GT_OR_RETURN(logical_shape->elem_cnt(), 0);
    const auto& in_nd_sbp = JUST(input_tensor->nd_sbp());
    const auto& out_nd_sbp = SymbolOf(JUST(MapAt(ndsbp_signature.bn_in_op2nd_sbp(), lbn)));
    const auto& in_parallel_desc = JUST(input_tensor->parallel_desc());
    const auto& out_parallel_desc = op_parallel_desc;
    CHECK_OR_RETURN(in_parallel_desc == out_parallel_desc);
    if (in_parallel_desc->parallel_num() != 1 && in_nd_sbp != out_nd_sbp) {
      const auto& boxing_interpreter = JUST(mgr->GetEagerBoxingInterpreter(
          in_nd_sbp, out_nd_sbp, in_parallel_desc, out_parallel_desc, *logical_shape));
      Singleton<const EagerBoxingLogger>::Get()->Log(
          *JUST(boxing_interpreter->boxing_interpreter_status()), /* prefix */ "");
      if (parallel_id.has_value()) {
        input_tensor = JUST(boxing_interpreter->Interpret(input_tensor, in_nd_sbp, out_nd_sbp,
                                                          in_parallel_desc, out_parallel_desc));
      }
      boxing_outputs.emplace_back(input_tensor);
    }
    const auto& local_tensor = JUST(input_tensor->cur_rank_phy_tensor());
    input_eager_blob_objects.emplace_back(JUST(local_tensor->eager_blob_object()));

    const auto& old_global_tensor_meta = infer_args.input_global_tensor_metas()[i].tensor_meta();
    GlobalTensorMeta global_tensor_meta(old_global_tensor_meta->shape(),
                                        old_global_tensor_meta->dtype(), in_nd_sbp,
                                        old_global_tensor_meta->parallel_desc());
    (*input_metas)[i] = SymbolOf(global_tensor_meta);
  }

  if (!parallel_id.has_value()) { return Maybe<void>::Ok(); }

  vm::EagerBlobObjectList output_eager_blob_objects(outputs->size());
  for (int i = 0; i < outputs->size(); ++i) {
    const bool out_parallel_desc_eq = JUST(outputs->at(i)->parallel_desc()) == parallel_desc;
    CHECK_OR_RETURN(out_parallel_desc_eq);
    const auto& local_tensor = JUST(outputs->at(i)->cur_rank_phy_tensor());
    output_eager_blob_objects.at(i) = JUST(local_tensor->eager_blob_object());
  }

  std::shared_ptr<const GlobalTensorInferResult> result =
      std::shared_ptr<const GlobalTensorInferResult>(std::move(mut_result));
  const auto& kernel = JUST(op->MutKernel4Stream(result->stream()));
  const auto& output_tensor_metas = result->output_tensor_metas();
  if (unlikely(JUST(CachedIsAllZeroSizeTensorMeta(output_tensor_metas)))) {
    return Maybe<void>::Ok();
  }

  JUST(PhysicalRun([&](InstructionsBuilder* builder) -> Maybe<void> {
    return builder->Call(kernel, std::move(input_eager_blob_objects),
                         std::move(output_eager_blob_objects), result, ctx, result->stream());
  }));

  for (size_t i = 0; i < output_names.size(); ++i) {
    env.emplace(output_names[i], JUST(VectorAt(*outputs, i)));
  }

  return Maybe<void>::Ok();
}

auto* RunGlobalNormalOpThenInitGlobalId = DECORATE(&RawRunGlobalNormalOp, NonRecursiveInitGlobalId);

}  // namespace

Maybe<void> RunGlobalNormalOp(const std::shared_ptr<UserOpExpr>& op, const TensorTuple& inputs,
                              Env& env, const OperatorConf& op_conf, const UserOpConf& user_conf,
                              const OpArgsVector<std::string>& ibns,
                              const OpArgsVector<std::string>& obns,
                              const OpArgsVector<std::string>& output_names,
                              const NdSbpSignature& ndsbp_signature,
                              const Symbol<ParallelDesc>& op_parallel_desc) {
  TensorTuple outputs(output_names.size());
  return RunGlobalNormalOpThenInitGlobalId(op, inputs, &outputs, env, op_conf, user_conf, ibns,
                                           obns, output_names, ndsbp_signature, op_parallel_desc);
}

Maybe<void> RunNormalOp(const std::shared_ptr<UserOpExpr>& op, Env& env, const TensorTuple& inputs,
                        const OpArgsVector<std::string>& output_names) {
  TensorTuple outputs(output_names.size());
  static EagerLocalInterpreter it;
  static AttrMap empty_attr_map;
  JUST(it.Apply(*op, inputs, &outputs, empty_attr_map));
  for (size_t i = 0; i < output_names.size(); ++i) {
    env.emplace(output_names[i], JUST(VectorAt(outputs, i)));
  }
  return Maybe<void>::Ok();
}

// tensors in outdated_tensors_after_op[i] will not be accessed any more after i-th op
// so they can be released once i-th op's execution finishes.
std::vector<std::vector<std::string>> GetOutdatedTensorsAfterOp(const Job& job) {
  std::vector<std::vector<std::string>> outdated_tensors_after_op(job.net().op_size());
  std::set<std::string> visited;
  for (int i = job.net().op_size() - 1; i >= 0; --i) {
    const auto& op_conf = job.net().op(i);
    // do not release the graph output tensors
    if (op_conf.has_output_conf()) {
      const auto& output_conf = op_conf.output_conf();
      visited.insert(output_conf.in());
    } else if (op_conf.has_user_conf()) {
      const auto& user_conf = op_conf.user_conf();
      for (const auto& pair : user_conf.input()) {
        if (pair.first == "UserSourceOpTickInput") { continue; }
        for (const auto& name : pair.second.s()) {
          if (visited.find(name) == visited.end()) {
            outdated_tensors_after_op[i].push_back(name);
            visited.insert(name);
          }
        }
      }
    }
  }
  return outdated_tensors_after_op;
}

Maybe<void> InitOpExprs(const std::shared_ptr<NNGraph>& graph) {
  CHECK_OR_RETURN(graph->cached_op_exprs.empty());

  const auto& job = graph->job();
  for (int i = 0; i < job.net().op_size(); i++) {
    const auto& op_conf = job.net().op(i);
    if (op_conf.has_user_conf()) {
      const auto op_expr = JUST(OpConfToUserOpExpr(op_conf));
      graph->cached_op_exprs.push_back(op_expr);
    } else {
      graph->cached_op_exprs.push_back(nullptr);
    }
  }
  return Maybe<void>::Ok();
}

Maybe<one::TensorTuple> InterpretJob(const one::TensorTuple& graph_inputs,
                                     const std::shared_ptr<NNGraph>& graph) {
  if (graph->cached_op_exprs.empty()) { JUST(InitOpExprs(graph)); }

  const auto& job = graph->job();
  auto env = *JUST(InitEnv(graph_inputs, graph));

  // See comments above GetOutdatedTensorsAfterOp's definition for more details
  const auto outdated_tensors_after_op = GetOutdatedTensorsAfterOp(job);

  CHECK_OR_RETURN(job.has_placement()) << "no job placement";
  const auto& job_placement = job.placement();
  NameToParallelDescType op2paralleldesc;
  for (const auto& blob_placement_group : job_placement.blob_placement_group()) {
    const auto parallel_desc = SymbolOf(ParallelDesc(blob_placement_group.parallel_conf()));
    for (const auto& logical_blob_id : blob_placement_group.lbi()) {
      op2paralleldesc.emplace(logical_blob_id.op_name(), parallel_desc);
    }
  }
  CHECK_OR_RETURN(job.has_job_parallel_view_conf()) << "no job parallel conf";
  const auto& job_parallel_view_conf = job.job_parallel_view_conf();
  const auto& op_name2nd_sbp_signature_conf =
      job_parallel_view_conf.op_name2nd_sbp_signature_conf();

  one::TensorTuple graph_outputs;
  for (int i = 0; i < job.net().op_size(); i++) {
    const auto& op_conf = job.net().op(i);
    if (op_conf.has_user_conf()) {
      auto op = CHECK_NOTNULL(graph->cached_op_exprs[i]);
      const auto& user_conf = op_conf.user_conf();
      OF_PROFILER_RANGE_GUARD(user_conf.op_type_name());
      const auto [inputs, ibns] =
          *JUST(GetInputTensors(user_conf, env, [&op_conf](const std::shared_ptr<Tensor>& tensor) {
            return CHECK_JUST(functional::To(tensor, op_conf.device_tag()));
          }));
      const auto [output_names, obns] = *JUST(GetOutputNamesOfOp(user_conf));
      if (JUST(GetEagerInterpreterType(inputs, op))) {
        if (IsViewOp(op)) {
          JUST(RunViewOp(op, env, inputs, output_names));
        } else {
          JUST(RunNormalOp(op, env, inputs, output_names));
        }
      } else {
        const auto& op_parallel_desc = JUST(MapAt(op2paralleldesc, op_conf.name()));
        const auto& nd_sbp_signature_conf =
            JUST(MapAt(op_name2nd_sbp_signature_conf, op_conf.name()));
        JUST(RunGlobalNormalOp(op, inputs, env, op_conf, user_conf, ibns, obns, output_names,
                               nd_sbp_signature_conf, op_parallel_desc));
      }
      for (const auto& name : outdated_tensors_after_op[i]) {
        CHECK_EQ_OR_RETURN(env.erase(name), 1);
      }
    } else if (op_conf.has_output_conf()) {
      const auto& output_conf = op_conf.output_conf();
      graph_outputs.emplace_back(JUST(MapAt(env, output_conf.in())));
    }
  }
  return graph_outputs;
}
}  // namespace one
}  // namespace oneflow
