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
#include "oneflow/core/register/runtime_blob_desc.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

std::function<bool(const OpNode* op_node)> MakePredicatorIsSafeToDelete(const OpGraph& op_graph) {
  HashSet<std::string> ctrl_in_op_names;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    for (const std::string& ctrl_in_op_name : op_node->op().op_conf().ctrl_in_op_name()) {
      ctrl_in_op_names.insert(ctrl_in_op_name);
    }
  });
  return [=](const OpNode* op_node) {
    if (op_node->out_edges().size() > 1) { return false; }
    if (!op_node->op().op_conf().ctrl_in_op_name().empty()) { return false; }
    if (ctrl_in_op_names.find(op_node->op().op_conf().name()) != ctrl_in_op_names.end()) {
      return false;
    }
    return true;
  };
}

bool IsUserOpWithTypeName(const OperatorConf& op_conf, const std::string& op_type_name) {
  return op_conf.has_user_conf() && op_conf.user_conf().op_type_name() == op_type_name;
};

class FuseCastScalePass final : public JobPass {
 public:
  FuseCastScalePass() = default;
  ~FuseCastScalePass() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ctx.job_desc().job_conf().enable_fuse_cast_scale();
  }
  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> FuseCastScalePass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  const auto IsSafeToDelete = MakePredicatorIsSafeToDelete(op_graph);
  op_graph.ForEachNode([&](const OpNode* op_node) {
    if (!IsUserOpWithTypeName(op_node->op().op_conf(), "cast")) { return; }
    if (!IsSafeToDelete(op_node)) { return; }
    if (op_node->out_edges().size() != 1) { return; }
    const OpNode* sole_dst_node = op_node->SoleOutEdge()->dst_node();
    if (!IsUserOpWithTypeName(sole_dst_node->op().op_conf(), "scalar_mul_by_tensor")) { return; }
    const user_op::UserOpConfWrapper cast_user_conf(op_node->op().op_conf());
    if (op_node->LogicalBlobDesc4Lbi(GenLogicalBlobId(cast_user_conf.input("in", 0))).data_type()
        != DataType::kFloat16) {
      return;
    }
    if (op_node->LogicalBlobDesc4Lbi(GenLogicalBlobId(cast_user_conf.output("out", 0))).data_type()
        != DataType::kFloat) {
      return;
    }
    if (op_node->parallel_desc().device_type() != DeviceType::kGPU) { return; }
    const user_op::UserOpConfWrapper scale_user_conf(sole_dst_node->op().op_conf());
    OperatorConf new_op_conf = sole_dst_node->op().op_conf();
    new_op_conf.mutable_user_conf()->set_op_type_name("fused_cast_scale");
    const auto new_val = cast_user_conf.input("in", 0);
    const auto& old_val =
        ReplaceInputLbnInOpCustomizedConf(&new_op_conf, GenRepeatedBn("x", 0), new_val);
    CHECK_EQ(scale_user_conf.input("x", 0), old_val);
    job_builder->DelOps({op_node->op().op_conf()});
    job_builder->MutOpsOnlyOnce({new_op_conf});
  });
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("FuseCastScalePass", FuseCastScalePass);

}  // namespace oneflow
