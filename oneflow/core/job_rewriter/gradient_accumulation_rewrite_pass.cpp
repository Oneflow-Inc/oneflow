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
#include "oneflow/core/common/multi_client.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

class GradientAccumulationRewritePass final : public JobPass {
 public:
  GradientAccumulationRewritePass() = default;
  ~GradientAccumulationRewritePass() override = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

Maybe<void> GradientAccumulationRewritePass::Apply(Job* job, JobPassCtx* ctx) const {
  const JobConfigProto& job_conf = ctx->job_desc().job_conf();
  if (!job_conf.has_train_conf()) { return Maybe<void>::Ok(); }
  if ((!job_conf.has_num_gradient_accumulation_steps())
      || job_conf.num_gradient_accumulation_steps() <= 1) {
    return Maybe<void>::Ok();
  }
  const bool is_multi_client = CHECK_JUST(IsMultiClient());
  if (is_multi_client) {
    // GradientAccumulationRewritePass has been re-implemented in op interpreter in multi client.
    return Maybe<void>::Ok();
  }
  const OpGraph op_graph(*job);
  JobBuilder job_builder(job);
  HashMap<std::string, OperatorConf> name2op_conf;
  auto GetOperatorConf4Modify = [&name2op_conf](const OperatorConf& op_conf) {
    const auto& it = name2op_conf.find(op_conf.name());
    if (it != name2op_conf.end()) {
      return &it->second;
    } else {
      name2op_conf[op_conf.name()] = op_conf;
      return &name2op_conf.at(op_conf.name());
    }
  };
  const int64_t repeat_num = GlobalJobDesc().job_conf().num_gradient_accumulation_steps();
  JUST(op_graph.TopoForEachNodeWithErrorCaptured([&](const OpNode* node) -> Maybe<void> {
    const OperatorConf& op_conf = node->op().op_conf();
    if (node->in_edges().empty()) {    // sources
      if (op_conf.has_input_conf()) {  // input
        // NOTE(chengcheng):
        //   We assume that the input data is one mini-batch which containing multi micro-batches.
        //   So we need unpack input data for each micro-batch.
        const LogicalBlobId input_lbi = node->op().BnInOp2Lbi("out");
        const std::string input_lbn = GenLogicalBlobName(input_lbi);

        user_op::UserOpConfWrapperBuilder unpack_builder("System-GradientAccumulation-InputUnpack-"
                                                         + op_conf.name() + "-" + NewUniqueId());
        const auto unpack_op = unpack_builder.OpTypeName("unpack")
                                   .Input("in", input_lbn)
                                   .Output("out")
                                   .Attr<int32_t>("unpack_num", repeat_num)
                                   .ScopeSymbolId(op_conf.scope_symbol_id())
                                   .Build();
        job_builder.AddOps(node->parallel_desc().parallel_conf(), {unpack_op.op_conf()});
        const std::string unpack_lbn = unpack_op.output("out", 0);
        node->ForEachNodeOnOutEdge([&](const OpNode* dst) {
          const auto& dst_op = dst->op();
          OperatorConf* new_dst_op_conf = GetOperatorConf4Modify(dst_op.op_conf());
          for (const auto& ibn : dst_op.input_bns()) {
            if (dst_op.BnInOp2Lbi(ibn) == input_lbi) {
              const auto& old_val =
                  ReplaceInputLbnInOpCustomizedConf(new_dst_op_conf, ibn, unpack_lbn);
              CHECK_EQ(input_lbn, old_val);
            }
          }
        });

        return Maybe<void>::Ok();
      } else if (op_conf.has_variable_conf()) {  // repeat variable
        const LogicalBlobId variable_lbi = node->op().BnInOp2Lbi("out");
        const std::string variable_lbn = GenLogicalBlobName(variable_lbi);
        HashMap<ParallelConf, std::string> parallel_conf2repeat_lbn;
        node->ForEachNodeOnOutEdge([&](const OpNode* dst) {
          const auto& dst_op = dst->op();
          const ParallelConf& parallel_conf = dst->parallel_desc().parallel_conf();
          std::string repeat_lbn;
          const auto& it = parallel_conf2repeat_lbn.find(parallel_conf);
          if (it == parallel_conf2repeat_lbn.end()) {
            user_op::UserOpConfWrapperBuilder repeat_builder(
                "System-GradientAccumulation-Repeat-" + op_conf.name() + "-" + NewUniqueId());
            const auto repeat_op = repeat_builder.OpTypeName("repeat")
                                       .Input("in", variable_lbn)
                                       .Output("out")
                                       .Attr<int32_t>("repeat_num", repeat_num)
                                       .ScopeSymbolId(dst_op.op_conf().scope_symbol_id())
                                       .Build();
            job_builder.AddOps(parallel_conf, {repeat_op.op_conf()});
            repeat_lbn = repeat_op.output("out", 0);
            parallel_conf2repeat_lbn.emplace(parallel_conf, repeat_lbn);
          } else {
            repeat_lbn = it->second;
          }
          OperatorConf* new_dst_op_conf = GetOperatorConf4Modify(dst_op.op_conf());
          for (const auto& ibn : dst_op.input_bns()) {
            if (dst_op.BnInOp2Lbi(ibn) == variable_lbi) {
              const auto& old_val =
                  ReplaceInputLbnInOpCustomizedConf(new_dst_op_conf, ibn, repeat_lbn);
              CHECK_EQ(variable_lbn, old_val);
            }
          }
        });
        return Maybe<void>::Ok();
      } else if (op_conf.has_user_conf()) {  // repeat tick
        OperatorConf* new_op_conf = GetOperatorConf4Modify(op_conf);
        OperatorConf tick_conf{};
        tick_conf.set_name("System-GradientAccumulation-RepeatTick-DeviceTick-" + op_conf.name());
        tick_conf.mutable_device_tick_conf()->set_out("out");
        tick_conf.set_scope_symbol_id(op_conf.scope_symbol_id());
        auto tick_lbn = GenLogicalBlobName(tick_conf.name(), tick_conf.device_tick_conf().out());
        user_op::UserOpConfWrapperBuilder repeat_builder(
            "System-GradientAccumulation-RepeatTick-Repeat-" + op_conf.name());
        const auto repeat_op = repeat_builder.OpTypeName("repeat")
                                   .Input("in", tick_lbn)
                                   .Output("out")
                                   .Attr<int32_t>("repeat_num", repeat_num)
                                   .ScopeSymbolId(op_conf.scope_symbol_id())
                                   .Build();
        job_builder.AddOps(node->parallel_desc().parallel_conf(), {tick_conf, repeat_op.op_conf()});
        (*new_op_conf->mutable_user_conf()->mutable_input())[user_op::kUserSourceOpTickInputArgName]
            .add_s(repeat_op.output("out", 0));
        return Maybe<void>::Ok();
      } else {
        LOG(ERROR) << "Gradient accumulation unsupported op : " << op_conf.DebugString();
        return Error::UnimplementedError();
      }
    } else if ((is_multi_client && op_conf.has_output_conf())
               || (!is_multi_client && op_conf.has_return_conf())) {
      // NOTE(chengcheng):
      //   in Single-Client GlobalFunction return op is output
      //   in Multi-Client nn.Graph output op is output.
      const LogicalBlobId return_in_lbi = node->op().BnInOp2Lbi("in");
      const std::string return_in_lbn = GenLogicalBlobName(return_in_lbi);
      user_op::UserOpConfWrapperBuilder pack_builder("System-GradientAccumulation-ReturnPack-"
                                                     + op_conf.name());
      const auto return_pack_op = pack_builder.OpTypeName("pack")
                                      .Input("in", return_in_lbn)
                                      .Output("out")
                                      .Attr<int32_t>("pack_num", repeat_num)
                                      .ScopeSymbolId(op_conf.scope_symbol_id())
                                      .Build();
      job_builder.AddOps(node->parallel_desc().parallel_conf(), {return_pack_op.op_conf()});
      OperatorConf* new_return_op_conf = GetOperatorConf4Modify(op_conf);
      const auto& old_val = ReplaceInputLbnInOpCustomizedConf(new_return_op_conf, "in",
                                                              return_pack_op.output("out", 0));
      CHECK_EQ(return_in_lbn, old_val);
      return Maybe<void>::Ok();
    } else if (is_multi_client && op_conf.has_user_conf()
               && op_conf.user_conf().op_type_name() == "reshape") {
      const LogicalBlobId in_lbi = node->op().BnInOp2Lbi(node->op().SoleIbn());
      const LogicalBlobId out_lbi = node->op().BnInOp2Lbi(node->op().SoleObn());
      const Shape& in_shape = node->LogicalBlobDesc4Lbi(in_lbi).shape();
      const Shape& out_shape = node->LogicalBlobDesc4Lbi(out_lbi).shape();
      if (in_shape.NumAxes() > 0 && out_shape.NumAxes() > 0 && in_shape.At(0) == out_shape.At(0)) {
        // NOTE(chengcheng):
        //  in nn.Graph GradientAccumulation, the reshape conf in JobBuild and after insert
        //  acc/unpack maybe NOT equal because of dim 0 scaled, so need set dim 0 as -1 for
        //  dynamic infer.
        OperatorConf* new_reshape_op_conf = GetOperatorConf4Modify(op_conf);
        AttrValue* attr_val = &(*new_reshape_op_conf->mutable_user_conf()->mutable_attr())["shape"];
        CHECK(attr_val->has_at_shape());
        ShapeProto* shape_conf = attr_val->mutable_at_shape();
        CHECK_GT(shape_conf->dim_size(), 0);
        shape_conf->set_dim(0, -1);
        LOG(INFO) << " Replace ReshapeOpConf from: " << op_conf.DebugString() << " to "
                  << new_reshape_op_conf->DebugString() << " for dynamic infer by insert unpack.";
      }
      return Maybe<void>::Ok();
    } else {
      return Maybe<void>::Ok();
    }
  }));
  for (const auto& pair : name2op_conf) { job_builder.MutOpsOnlyOnce({pair.second}); }
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("GradientAccumulationRewritePass", GradientAccumulationRewritePass);

}  // namespace oneflow
