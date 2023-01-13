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
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

class DelayVariableOpExecutionPass final : public JobPass {
 public:
  DelayVariableOpExecutionPass() = default;
  ~DelayVariableOpExecutionPass() override = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

Maybe<void> DelayVariableOpExecutionPass::Apply(Job* job, JobPassCtx* ctx) const {
  if (!ParseBooleanFromEnv("ONEFLOW_GRAPH_DELAY_VARIABLE_OP_EXECUTION", false)) {
    return Maybe<void>::Ok();
  }
  const JobConfigProto& job_conf = ctx->job_desc().job_conf();
  if (job_conf.has_train_conf()) { return Maybe<void>::Ok(); }
  if (job_conf.has_num_gradient_accumulation_steps()
      && job_conf.num_gradient_accumulation_steps() > 1) {
    return Maybe<void>::Ok();
  }
  if (GlobalProcessCtx::WorldSize() > 1) { return Maybe<void>::Ok(); }
  const OpGraph op_graph(*job);
  JobBuilder job_builder(job);
  JUST(op_graph.TopoForEachNodeWithErrorCaptured([&](const OpNode* node) -> Maybe<void> {
    const OperatorConf& op_conf = node->op().op_conf();
    if (!op_conf.has_variable_conf()) { return Maybe<void>::Ok(); }
    if (!op_conf.ctrl_in_op_name().empty()) { return Maybe<void>::Ok(); }
    if (op_conf.variable_conf().has_tick()) { return Maybe<void>::Ok(); }
    if (node->out_edges().size() != 1) { return Maybe<void>::Ok(); }
    if (node->parallel_desc().parallel_num() != 1) { return Maybe<void>::Ok(); }
    const OpNode* dst_node = (*node->out_edges().begin())->dst_node();
    if (dst_node->parallel_desc() != node->parallel_desc()) { return Maybe<void>::Ok(); }

    const OpEdge* none_variable_edge = nullptr;
    for (const OpEdge* edge : dst_node->in_edges()) {
      if (edge->src_node()->op().op_conf().has_variable_conf()) { continue; }
      if (edge->lbis().size() == 0) { continue; }
      if (edge->src_node()->parallel_desc() != node->parallel_desc()) { continue; }
      none_variable_edge = edge;
      break;
    }
    if (none_variable_edge == nullptr) { return Maybe<void>::Ok(); }
    OperatorConf new_varibale_conf = op_conf;
    new_varibale_conf.mutable_variable_conf()->set_tick(
        GenLogicalBlobName(none_variable_edge->lbis().front()));
    job_builder.MutOpsOnlyOnce({new_varibale_conf});
    return Maybe<void>::Ok();
  }));
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("DelayVariableOpExecutionPass", DelayVariableOpExecutionPass);

}  // namespace oneflow
