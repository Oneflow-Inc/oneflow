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
#include "oneflow/core/common/util.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

bool IsUserOpWithTypeName(const OperatorConf& op_conf, const std::string& op_type_name) {
  return op_conf.has_user_conf() && op_conf.user_conf().op_type_name() == op_type_name;
};

class MultiTensorUpdatePass final : public JobPass {
 public:
  MultiTensorUpdatePass() = default;
  ~MultiTensorUpdatePass() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ParseBooleanFromEnv("ONEFLOW_ENABLE_MULTI_TENSOR_UPDATE", false);
  }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> MultiTensorUpdatePass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  op_graph.ForEachNode([&](OpNode* op_node) {
    const auto& op_conf = op_node->op().op_conf();
    if (!op_conf.has_variable_conf()) { return; }
    LogicalBlobId model_half_lbi;

    for (OpEdge* find_sgd_edge : op_node->out_edges()) {
      OpNode* find_sgd_update_node = find_sgd_edge->dst_node();
      if (!IsUserOpWithTypeName(find_sgd_update_node->op().op_conf(), "sgd_update")) { continue; }
      const user_op::UserOpConfWrapper sgd_user_conf(find_sgd_update_node->op().op_conf());
      // Currently only support for cuda, maybe remove this limit.
      if (find_sgd_update_node->parallel_desc().device_type() != DeviceType::kCUDA) { continue; }

      user_op::UserOpConfWrapperBuilder fused_sgd_op_builder(sgd_user_conf.op_name());
      fused_sgd_op_builder.OpTypeName("multi_tensor_sgd_update")
          .Input("model", sgd_user_conf.input("model", 0))
          .Input("model_diff", sgd_user_conf.input("model_diff", 0))
          .Input("learning_rate", sgd_user_conf.input("learning_rate", 0))
          .Attr<double>("scale", sgd_user_conf.attr<double>("scale"))
          .Attr<float>("l1", sgd_user_conf.attr<float>("l1"))
          .Attr<float>("l2", sgd_user_conf.attr<float>("l2"))
          .Attr<float>("weight_decay", sgd_user_conf.attr<float>("weight_decay"));

      CHECK(sgd_user_conf.op_conf().has_scope_symbol_id());
      fused_sgd_op_builder.ScopeSymbolId(sgd_user_conf.op_conf().scope_symbol_id());

      OperatorConf new_sgd_op_conf = sgd_user_conf.op_conf();
      *new_sgd_op_conf.mutable_user_conf() = fused_sgd_op_builder.Build().op_conf().user_conf();
      job_builder->MutOpsOnlyOnce({new_sgd_op_conf});
      break;
    }
  });
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("MultiTensorUpdatePass", MultiTensorUpdatePass);

}  // namespace oneflow
