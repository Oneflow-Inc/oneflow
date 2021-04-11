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

class IndexedSlicesOptimizerRewritePass final : public JobPass {
 public:
  IndexedSlicesOptimizerRewritePass() = default;
  ~IndexedSlicesOptimizerRewritePass() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ctx.job_desc().job_conf().has_indexed_slices_optimizer_conf()
           && ctx.job_desc().job_conf().indexed_slices_optimizer_conf().enable();
  }

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> IndexedSlicesOptimizerRewritePass::Apply(const OpGraph& op_graph,
                                                     JobBuilder* job_builder) const {
  const PbRpf<std::string>& include_op_names =
      GlobalJobDesc().job_conf().indexed_slices_optimizer_conf().include_op_names().op_name();
  const std::set<std::string> include_op_name_set(
      {include_op_names.cbegin(), include_op_names.cend()});
  op_graph.ForEachNode([&](const OpNode* src_node) {
    const OperatorConf& src_op_conf = src_node->op().op_conf();
    if (src_node->out_edges().size() != 1) { return; }
    std::string indices_lbn;
    std::string values_lbn;
    std::string model_op_name;
    if (!src_op_conf.has_user_conf()) { return; }
    const user_op::UserOpConfWrapper src_op(src_op_conf);
    if (src_op.op_type_name() == "unsorted_segment_sum" && src_op.attr<int64_t>("axis") == 0) {
      indices_lbn = src_op.input("segment_ids", 0);
      values_lbn = src_op.input("data", 0);
    } else if (src_op.op_type_name() == "unsorted_segment_sum_like"
               && src_op.attr<int64_t>("axis") == 0) {
      indices_lbn = src_op.input("segment_ids", 0);
      values_lbn = src_op.input("data", 0);
    } else {
      return;
    }
    std::vector<const OpNode*> op_nodes_to_remove;
    std::vector<const OpNode*> op_nodes_apply_to_diff;
    const OpNode* dst_node = src_node->SoleOutEdge()->dst_node();
    do {
      if (dst_node->op().output_bns().empty()) { break; }
      const OperatorConf& dst_op_conf = dst_node->op().op_conf();
      if (dst_op_conf.has_user_conf()
          && dst_op_conf.user_conf().op_type_name() == "hierarchical_parallel_cast") {
        if (dst_node->out_edges().size() != 1) { return; }
        op_nodes_to_remove.push_back(dst_node);
        dst_node = dst_node->SoleOutEdge()->dst_node();
        continue;
      } else if (dst_op_conf.has_user_conf()
                 && dst_op_conf.user_conf().op_type_name() == "scalar_mul") {
        if (dst_node->out_edges().size() != 1) { return; }
        op_nodes_apply_to_diff.push_back(dst_node);
        dst_node = dst_node->SoleOutEdge()->dst_node();
        continue;
      } else {
        return;
      }
    } while (true);
    if (!dst_node->op().op_conf().has_user_conf()) { return; }
    const user_op::UserOpConfWrapper user_op_conf(dst_node->op().op_conf());
    if (user_op_conf.op_type_name() != "sgd_update"
        && user_op_conf.op_type_name() != "momentum_update"
        && user_op_conf.op_type_name() != "adam_update") {
      return;
    }
    if (user_op_conf.attr<double>("scale") != 1.0 || user_op_conf.attr<float>("l1") != 0.0f
        || user_op_conf.attr<float>("l2") != 0.0f || user_op_conf.has_input("scale_by_tensor", 0)) {
      return;
    }
    const LogicalBlobId& model_lbi = GenLogicalBlobId(user_op_conf.input("model", 0));
    if (dst_node->LogicalBlobDesc4Lbi(GenLogicalBlobId(user_op_conf.input("model_diff", 0)))
            .data_type()
        != dst_node->LogicalBlobDesc4Lbi(model_lbi).data_type()) {
      return;
    }
    model_op_name = model_lbi.op_name();
    user_op::UserOpConfWrapperBuilder indexed_slices_op_builder("System-Optimizer-IndexedSlices-"
                                                                + model_op_name);
    indexed_slices_op_builder.OpTypeName("indexed_slices_" + user_op_conf.op_type_name())
        .Input("model", user_op_conf.input("model", 0))
        .Input("learning_rate", user_op_conf.input("learning_rate", 0))
        .Attr<float>("weight_decay", user_op_conf.attr<float>("weight_decay"));

    if (user_op_conf.op_type_name() == "sgd_update") {
      // do nothing
    } else if (user_op_conf.op_type_name() == "momentum_update") {
      indexed_slices_op_builder.Input("momentum", user_op_conf.input("momentum", 0))
          .Attr<float>("beta", user_op_conf.attr<float>("beta"));
    } else if (user_op_conf.op_type_name() == "adam_update") {
      indexed_slices_op_builder.Input("m", user_op_conf.input("m", 0))
          .Input("v", user_op_conf.input("v", 0))
          .Attr<float>("beta1", user_op_conf.attr<float>("beta1"))
          .Attr<float>("beta2", user_op_conf.attr<float>("beta2"))
          .Attr<float>("epsilon", user_op_conf.attr<float>("epsilon"));
    } else {
      return;
    }
    CHECK(!model_op_name.empty());
    CHECK(!indices_lbn.empty());
    CHECK(!values_lbn.empty());
    if (include_op_name_set.find(model_op_name) == include_op_name_set.end()) { return; }
    for (const OpNode* node : op_nodes_to_remove) { job_builder->DelOps({node->op().op_conf()}); }
    for (const OpNode* node : op_nodes_apply_to_diff) {
      OperatorConf new_conf = node->op().op_conf();
      if (new_conf.has_user_conf() && new_conf.user_conf().op_type_name() == "scalar_mul") {
        const auto& old_val = ReplaceInputLbnInOpCustomizedConf(&new_conf, "in_0", values_lbn);
        CHECK_EQ(GenLogicalBlobName(node->op().BnInOp2Lbi("in_0")), old_val);
        values_lbn = GenLogicalBlobName(new_conf.name(), "out_0");
        job_builder->MutOpsOnlyOnce({new_conf});
      } else {
        UNIMPLEMENTED();
      }
    }
    indexed_slices_op_builder.Input("model_diff_indices", indices_lbn)
        .Input("model_diff_values", values_lbn);
    job_builder->DelOps({src_op_conf, user_op_conf.op_conf()});
    job_builder->AddOps(dst_node->parallel_desc().parallel_conf(),
                        {indexed_slices_op_builder.Build().op_conf()});
  });
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("IndexedSlicesOptimizerRewritePass", IndexedSlicesOptimizerRewritePass);

}  // namespace oneflow
