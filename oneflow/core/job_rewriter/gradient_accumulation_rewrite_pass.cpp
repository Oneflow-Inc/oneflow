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

class InsertGradAccForwardPlaceholderPass final : public JobPass {
 public:
  InsertGradAccForwardPlaceholderPass() = default;
  ~InsertGradAccForwardPlaceholderPass() override = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

Maybe<void> InsertGradAccForwardPlaceholderPass::Apply(Job* job, JobPassCtx* ctx) const {
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
  const int64_t acc_num = GlobalJobDesc().job_conf().num_gradient_accumulation_steps();
  JUST(op_graph.TopoForEachNodeWithErrorCaptured([&](const OpNode* node) -> Maybe<void> {
    const OperatorConf& op_conf = node->op().op_conf();
    if (op_conf.has_variable_conf()) {
      // NOTE(chengcheng):
      //  ONLY need insert _grad_acc_forward_placeholder op after variable
      const LogicalBlobId variable_lbi = node->op().BnInOp2Lbi("out");
      const std::string variable_lbn = GenLogicalBlobName(variable_lbi);
      const ParallelConf& parallel_conf = node->parallel_desc().parallel_conf();
      user_op::UserOpConfWrapperBuilder grad_acc_fw_ph_builder(
          "System-GradientAccumulation-FwPlaceholder-" + op_conf.name());
      const auto grad_acc_placeholder_op =
          grad_acc_fw_ph_builder.OpTypeName("_grad_acc_forward_placeholder")
              .Input("in", variable_lbn)
              .Output("out")
              .Attr<int32_t>("acc_num", acc_num)
              .ScopeSymbolId(op_conf.scope_symbol_id())
              .Build();

      job_builder.AddOps(parallel_conf, {grad_acc_placeholder_op.op_conf()});
      const std::string new_var_lbn = grad_acc_placeholder_op.output("out", 0);

      node->ForEachNodeOnOutEdge([&](const OpNode* dst) {
        const auto& dst_op = dst->op();
        OperatorConf* new_dst_op_conf = GetOperatorConf4Modify(dst_op.op_conf());
        for (const auto& ibn : dst_op.input_bns()) {
          if (dst_op.BnInOp2Lbi(ibn) == variable_lbi) {
            const auto& old_val =
                ReplaceInputLbnInOpCustomizedConf(new_dst_op_conf, ibn, new_var_lbn);
            CHECK_EQ(variable_lbn, old_val);
          }
        }
      });
    }
    return Maybe<void>::Ok();
  }));
  for (const auto& pair : name2op_conf) { job_builder.MutOpsOnlyOnce({pair.second}); }
  return Maybe<void>::Ok();
}

class PruneGradAccForwardPlaceholderPass final : public JobPass {
 public:
  PruneGradAccForwardPlaceholderPass() = default;
  ~PruneGradAccForwardPlaceholderPass() override = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

Maybe<void> PruneGradAccForwardPlaceholderPass::Apply(Job* job, JobPassCtx* ctx) const {
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
  std::vector<std::string> del_op_names;
  JUST(op_graph.TopoForEachNodeWithErrorCaptured([&](const OpNode* node) -> Maybe<void> {
    const OperatorConf& op_conf = node->op().op_conf();
    if (op_conf.has_user_conf()
        && op_conf.user_conf().op_type_name() == "_grad_acc_forward_placeholder"
        && op_conf.ctrl_in_op_name().empty()) {
      // NOTE(chengcheng): Remove placeholder op after autograd.
      const user_op::UserOpConfWrapper fw_ph_op(op_conf);
      const std::string in_lbn = fw_ph_op.input("in", 0);
      const std::string out_lbn = fw_ph_op.output("out", 0);
      const LogicalBlobId& out_lbi = GenLogicalBlobId(out_lbn);

      node->ForEachNodeOnOutEdge([&](const OpNode* dst) {
        const auto& dst_op = dst->op();
        OperatorConf* new_dst_op_conf = GetOperatorConf4Modify(dst_op.op_conf());
        for (const auto& ibn : dst_op.input_bns()) {
          if (dst_op.BnInOp2Lbi(ibn) == out_lbi) {
            const auto& old_val = ReplaceInputLbnInOpCustomizedConf(new_dst_op_conf, ibn, in_lbn);
            CHECK_EQ(out_lbn, old_val);
          }
        }
      });
      del_op_names.emplace_back(op_conf.name());
    }
    return Maybe<void>::Ok();
  }));
  for (const auto& pair : name2op_conf) { job_builder.MutOpsOnlyOnce({pair.second}); }
  job_builder.DelOps(del_op_names);
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("InsertGradAccForwardPlaceholderPass", InsertGradAccForwardPlaceholderPass);
REGISTER_JOB_PASS("PruneGradAccForwardPlaceholderPass", PruneGradAccForwardPlaceholderPass);

}  // namespace oneflow
