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
    if (node->in_edges().empty()) {       // sources
      if (op_conf.has_variable_conf()) {  // repeat variable
        const LogicalBlobId variable_lbi = node->op().BnInOp2Lbi("out");
        const std::string variable_lbn = GenLogicalBlobName(variable_lbi);
        HashMap<ParallelConf, std::string> parallel_conf2repeat_lbn;
        node->ForEachNodeOnOutEdge([&](const OpNode* dst) {
          const ParallelConf& parallel_conf = dst->parallel_desc().parallel_conf();
          if (parallel_conf2repeat_lbn.find(parallel_conf) == parallel_conf2repeat_lbn.end()) {
            user_op::UserOpConfWrapperBuilder repeat_builder(
                "System-GradientAccumulation-Repeat-" + op_conf.name() + "-" + NewUniqueId());
            const auto repeat_op = repeat_builder.OpTypeName("repeat")
                                       .Input("in", variable_lbn)
                                       .Output("out")
                                       .Attr<int32_t>("repeat_num", repeat_num)
                                       .ScopeSymbolId(dst->op().op_conf().scope_symbol_id())
                                       .Build();
            job_builder.AddOps(parallel_conf, {repeat_op.op_conf()});
            parallel_conf2repeat_lbn.emplace(parallel_conf, repeat_op.output("out", 0));
          }
        });
        node->ForEachNodeOnOutEdge([&](const OpNode* dst) {
          const auto& dst_op = dst->op();
          OperatorConf* new_dst_op_conf = GetOperatorConf4Modify(dst_op.op_conf());
          const std::string& repeat_lbn =
              parallel_conf2repeat_lbn.at(dst->parallel_desc().parallel_conf());
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
        tick_conf.set_name("System-GradientAccumulation-RepeatTick-Tick-" + op_conf.name());
        tick_conf.mutable_tick_conf()->set_out("out");
        tick_conf.set_scope_symbol_id(op_conf.scope_symbol_id());
        user_op::UserOpConfWrapperBuilder repeat_builder(
            "System-GradientAccumulation-RepeatTick-Repeat-" + op_conf.name());
        const auto repeat_op =
            repeat_builder.OpTypeName("repeat")
                .Input("in", GenLogicalBlobName(tick_conf.name(), tick_conf.tick_conf().out()))
                .Output("out")
                .Attr<int32_t>("repeat_num", repeat_num)
                .ScopeSymbolId(op_conf.scope_symbol_id())
                .Build();
        job_builder.AddOps(node->parallel_desc().parallel_conf(), {tick_conf, repeat_op.op_conf()});
        (*new_op_conf->mutable_user_conf()->mutable_input())[user_op::kUserSourceOpTickInputArgName]
            .add_s(repeat_op.output("out", 0));
        return Maybe<void>::Ok();
      } else {
        return Error::Unimplemented();
      }
    } else if (op_conf.has_return_conf()) {  // pack return
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
    } else {
      return Maybe<void>::Ok();
    }
  }));
  for (const auto& pair : name2op_conf) { job_builder.MutOpsOnlyOnce({pair.second}); }
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("GradientAccumulationRewritePass", GradientAccumulationRewritePass);

}  // namespace oneflow
