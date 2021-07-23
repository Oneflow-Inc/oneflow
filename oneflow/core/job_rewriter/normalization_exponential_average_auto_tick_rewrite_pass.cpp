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

class NormalizationExponentialAverageAutoTickPass final : public JobPass {
 public:
  NormalizationExponentialAverageAutoTickPass() = default;
  ~NormalizationExponentialAverageAutoTickPass() override = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

Maybe<void> NormalizationExponentialAverageAutoTickPass::Apply(Job* job, JobPassCtx* ctx) const {
  const JobConfigProto& job_conf = ctx->job_desc().job_conf();
  if (!job_conf.has_train_conf()) { return Maybe<void>::Ok(); }
  if ((!job_conf.has_num_gradient_accumulation_steps())
      || job_conf.num_gradient_accumulation_steps() <= 1) {
    return Maybe<void>::Ok();
  }
  const OpGraph op_graph(*job);
  JobBuilder job_builder(job);
  JUST(op_graph.TopoForEachNodeWithErrorCaptured([&](const OpNode* node) -> Maybe<void> {
    const OperatorConf& op_conf = node->op().op_conf();
    if (!op_conf.has_user_conf()) { return Maybe<void>::Ok(); }
    const user_op::UserOpConfWrapper user_op_conf(op_conf);
    if (user_op_conf.op_type_name() != "normalization"
        && user_op_conf.op_type_name() != "normalization_add_relu") {
      return Maybe<void>::Ok();
    }
    const std::string& x_lbn = user_op_conf.input("x", 0);
    const std::string& moving_mean_lbn = user_op_conf.input("moving_mean", 0);
    const std::string& moving_variance_lbn = user_op_conf.input("moving_variance", 0);
    std::string x_tick_lbn;
    auto GetXTick = [&]() {
      if (x_tick_lbn.empty()) {
        user_op::UserOpConfWrapperBuilder cast_to_tick_builder("System-CastToTick-"
                                                               + NewUniqueId());
        const auto cast_to_tick_op = cast_to_tick_builder.OpTypeName("cast_to_tick")
                                         .Input("in", x_lbn)
                                         .Output("out")
                                         .Build();
        job_builder.AddOps(node->parallel_desc().parallel_conf(), {cast_to_tick_op.op_conf()});
        x_tick_lbn = cast_to_tick_op.output("out", 0);
      }
      return x_tick_lbn;
    };
    auto TrySetTickForNode = [&](const OpNode* var_node) {
      if (!var_node->in_edges().empty()) { return; }
      if (!var_node->op().op_conf().has_variable_conf()) { return; }
      if (var_node->op().op_conf().variable_conf().has_tick()) { return; }
      OperatorConf new_var_op_conf = var_node->op().op_conf();
      new_var_op_conf.mutable_variable_conf()->set_tick(GetXTick());
      job_builder.MutOpsOnlyOnce({new_var_op_conf});
    };
    TrySetTickForNode(op_graph.OpNode4OpName(GenLogicalBlobId(moving_mean_lbn).op_name()));
    TrySetTickForNode(op_graph.OpNode4OpName(GenLogicalBlobId(moving_variance_lbn).op_name()));
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("NormalizationExponentialAverageAutoTickPass",
                  NormalizationExponentialAverageAutoTickPass);

}  // namespace oneflow
