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

namespace {

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
  std::vector<OperatorConf> delete_ops;
  op_graph.ForEachNode([&](const OpNode* op_node) {
    if (!IsUserOpWithTypeName(op_node->op().op_conf(), "cast")) { return; }
    if (!IsSafeToDelete(op_node)) { return; }
    if (op_node->out_edges().size() != 1) { return; }
    OpNode* sole_dst_node = op_node->SoleOutEdge()->dst_node();
    if (IsUserOpWithTypeName(sole_dst_node->op().op_conf(), "scalar_mul")) {
      if (!IsSafeToDelete(sole_dst_node)) { return; }
      if (!IsUserOpWithTypeName(sole_dst_node->SoleOutEdge()->dst_node()->op().op_conf(),
                                "scalar_mul_by_tensor")) {
        return;
      }
    } else {
      if (!IsUserOpWithTypeName(sole_dst_node->op().op_conf(), "scalar_mul_by_tensor")) { return; }
    }
    const user_op::UserOpConfWrapper cast_user_conf(op_node->op().op_conf());
    if (op_node->LogicalBlobDesc4Lbi(GenLogicalBlobId(cast_user_conf.input("in", 0))).data_type()
            != DataType::kFloat16
        && op_node->LogicalBlobDesc4Lbi(GenLogicalBlobId(cast_user_conf.input("in", 0))).data_type()
               != DataType::kBFloat16) {
      return;
    }
    if (op_node->LogicalBlobDesc4Lbi(GenLogicalBlobId(cast_user_conf.output("out", 0))).data_type()
        != DataType::kFloat) {
      return;
    }
    if (op_node->parallel_desc().device_type() != DeviceType::kCUDA) { return; }
    double scale = 1.0;
    if (IsUserOpWithTypeName(sole_dst_node->op().op_conf(), "scalar_mul")) {
      const user_op::UserOpConfWrapper scalar_mul_op_conf(sole_dst_node->op().op_conf());
      if (scalar_mul_op_conf.attr<bool>("has_int_operand")) {
        scale = static_cast<double>(scalar_mul_op_conf.attr<int64_t>("int_operand"));
      } else if (scalar_mul_op_conf.attr<bool>("has_float_operand")) {
        scale = scalar_mul_op_conf.attr<double>("float_operand");
      } else {
        UNIMPLEMENTED();
      }
      delete_ops.emplace_back(sole_dst_node->op().op_conf());
      sole_dst_node = sole_dst_node->SoleOutEdge()->dst_node();
    }
    delete_ops.emplace_back(op_node->op().op_conf());
    const user_op::UserOpConfWrapper scale_user_conf(sole_dst_node->op().op_conf());

    user_op::UserOpConfWrapperBuilder fused_op_builder(sole_dst_node->op().op_name());
    fused_op_builder.OpTypeName("fused_cast_scale")
        .Input("x", cast_user_conf.input("in", 0))
        .Input("scale_by_tensor", scale_user_conf.input("scalar", 0))
        .Attr<double>("scale", scale)
        .Output("y");

    OperatorConf new_op_conf = sole_dst_node->op().op_conf();
    *new_op_conf.mutable_user_conf() = fused_op_builder.Build().op_conf().user_conf();

    job_builder->MutOpsOnlyOnce({new_op_conf});
  });
  job_builder->DelOps(delete_ops);
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("FuseCastScalePass", FuseCastScalePass);

}  // namespace oneflow
