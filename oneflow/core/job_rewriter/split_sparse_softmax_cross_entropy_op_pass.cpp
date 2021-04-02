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

void UpdateProbConsumerOpConf(const std::string& new_prob_lbn, const OpNode* op_node,
                              JobBuilder* job_builder) {
  for (const OpEdge* edge : op_node->out_edges()) {
    OpNode* out_node = edge->dst_node();
    OperatorConf new_conf = out_node->op().op_conf();
    if (new_conf.has_user_conf()
        && new_conf.user_conf().op_type_name() == "sparse_softmax_cross_entropy_ms_grad") {
      CHECK_EQ(GenLogicalBlobName(out_node->op().BnInOp2Lbi("prob_0")),
               ReplaceInputLbnInOpCustomizedConf(&new_conf, "prob_0", new_prob_lbn));
      job_builder->MutOpsOnlyOnce({new_conf});
    }
  }
}

class SplitSparseSoftmaxCrossEntropyOpPass final : public JobPass {
 public:
  SplitSparseSoftmaxCrossEntropyOpPass() = default;
  ~SplitSparseSoftmaxCrossEntropyOpPass() override = default;

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> SplitSparseSoftmaxCrossEntropyOpPass::Apply(const OpGraph& op_graph,
                                                        JobBuilder* job_builder) const {
  op_graph.ForEachNode([&](const OpNode* node) {
    const OperatorConf& op_conf = node->op().op_conf();
    if (!op_conf.has_user_conf()) { return; }
    if (op_conf.user_conf().op_type_name() != "sparse_softmax_cross_entropy_ms") { return; }

    const int64_t scope_symbol_id = node->op().op_conf().scope_symbol_id();
    user_op::UserOpConfWrapper cur_op(op_conf);
    const std::string op_prediction_blob_name = cur_op.input("prediction", 0);
    const std::string op_label_blob_name = cur_op.input("label", 0);
    const int64_t depth = cur_op.attr<int64_t>("depth");
    const int32_t split_axis =
        node->LogicalBlobDesc4Lbi(node->op().BnInOp2Lbi("prediction_0")).shape().NumAxes() - 1;
    const std::vector<int32_t> axis_vec(1, split_axis);

    std::string op_name = node->op().op_name();
    const auto& op_parallel_distribution_sig =
        job_builder->ParallelDistributionSignature4OpName(op_name);
    const auto& parallel_distribution_map =
        op_parallel_distribution_sig.bn_in_op2parallel_distribution();
    const auto it = parallel_distribution_map.find("prediction_0");
    CHECK(it != parallel_distribution_map.end());
    const auto& prediction_parallel_distribution = it->second;

    ParallelDistribution global_max_or_sum_parallel_distribution;

    bool has_split_axis_parallel = false;
    CHECK_EQ(prediction_parallel_distribution.sbp_parallel_size(),
             node->parallel_desc().hierarchy()->NumAxes());
    for (int64_t i = 0; i < node->parallel_desc().hierarchy()->NumAxes(); ++i) {
      const auto& sbp = prediction_parallel_distribution.sbp_parallel(i);
      if (sbp.has_split_parallel() && sbp.split_parallel().axis() == split_axis) {
        has_split_axis_parallel = true;
        global_max_or_sum_parallel_distribution.add_sbp_parallel()->mutable_broadcast_parallel();
      } else {
        CHECK(!sbp.has_partial_sum_parallel());
        *global_max_or_sum_parallel_distribution.add_sbp_parallel() = sbp;
      }
    }
    CHECK(has_split_axis_parallel);

    auto reduce_max_device_stage_op =
        user_op::UserOpConfWrapperBuilder(op_name + "-split_softmax_reduce_max_device_stage")
            .Op("reduce_max_device_stage")
            .Input("in", op_prediction_blob_name)
            .Output("out")
            .Output("mask")
            .Output("count")
            .Attr("axis", axis_vec)
            .Build();
    job_builder->AddOps(node->parallel_desc().parallel_conf(),
                        {reduce_max_device_stage_op.op_conf()});
    ParallelDistributionSignature reduce_max_device_stage_signature;
    (*reduce_max_device_stage_signature.mutable_bn_in_op2parallel_distribution())["in_0"] =
        prediction_parallel_distribution;
    (*reduce_max_device_stage_signature.mutable_bn_in_op2parallel_distribution())["out_0"] =
        prediction_parallel_distribution;
    (*reduce_max_device_stage_signature.mutable_bn_in_op2parallel_distribution())["mask_0"] =
        prediction_parallel_distribution;
    (*reduce_max_device_stage_signature.mutable_bn_in_op2parallel_distribution())["count_0"] =
        prediction_parallel_distribution;
    job_builder->AddParallelDistributionSignature4OpName(reduce_max_device_stage_op.op_name(),
                                                         reduce_max_device_stage_signature);

    auto reduce_max_global_stage_op =
        user_op::UserOpConfWrapperBuilder(op_name + "-split_softmax_reduce_max_global_stage")
            .Op("reduce_max_global_stage")
            .Input("in", reduce_max_device_stage_op.output("out", 0))
            .Input("device_count", reduce_max_device_stage_op.output("count", 0))
            .Output("out")
            .Output("mask")
            .Attr("axis", axis_vec)
            .Attr("keepdims", true)
            .Build();
    job_builder->AddOps(node->parallel_desc().parallel_conf(),
                        {reduce_max_global_stage_op.op_conf()});
    ParallelDistributionSignature reduce_max_global_stage_signature;
    (*reduce_max_global_stage_signature.mutable_bn_in_op2parallel_distribution())["in_0"] =
        global_max_or_sum_parallel_distribution;
    (*reduce_max_global_stage_signature
          .mutable_bn_in_op2parallel_distribution())["device_count_0"] =
        global_max_or_sum_parallel_distribution;
    (*reduce_max_global_stage_signature.mutable_bn_in_op2parallel_distribution())["out_0"] =
        global_max_or_sum_parallel_distribution;
    job_builder->AddParallelDistributionSignature4OpName(reduce_max_global_stage_op.op_name(),
                                                         reduce_max_global_stage_signature);

    auto broadcast_sub_max_op =
        user_op::UserOpConfWrapperBuilder(op_name + "-split_softmax_sub_max")
            .Op("broadcast_sub")
            .Input("x", op_prediction_blob_name)
            .Input("y", reduce_max_global_stage_op.output("out", 0))
            .Output("z")
            .Build();
    job_builder->AddOps(node->parallel_desc().parallel_conf(), {broadcast_sub_max_op.op_conf()});

    auto exp_op = user_op::UserOpConfWrapperBuilder(op_name + "-split_softmax_exp")
                      .Op("exp")
                      .Input("x", broadcast_sub_max_op.output("z", 0))
                      .Output("y")
                      .Build();
    job_builder->AddOps(node->parallel_desc().parallel_conf(), {exp_op.op_conf()});

    auto reduce_sum_op = user_op::UserOpConfWrapperBuilder(op_name + "-split_softmax_reduce_sum")
                             .Op("reduce_sum")
                             .Input("input_tensor", exp_op.output("y", 0))
                             .Output("output_tensor")
                             .Attr("axis", axis_vec)
                             .Attr("keepdims", true)
                             .Build();
    job_builder->AddOps(node->parallel_desc().parallel_conf(), {reduce_sum_op.op_conf()});

    std::string reduce_sum_op_out = reduce_sum_op.output("output_tensor", 0);
    if (node->parallel_desc().hierarchy()->NumAxes() > 1) {
      std::vector<std::string> parallel_distribution_conf;
      for (const auto& sbp_parallel : global_max_or_sum_parallel_distribution.sbp_parallel()) {
        parallel_distribution_conf.push_back(SbpParallelToString(sbp_parallel));
      }
      auto parallel_cast_sum_op =
          user_op::UserOpConfWrapperBuilder(op_name + "-split_softmax_reduce_sum_cast_P2B")
              .Op("hierarchical_parallel_cast")
              .Input("in", reduce_sum_op.output("output_tensor", 0))
              .Output("out")
              .Attr<std::vector<std::string>>("parallel_distribution", parallel_distribution_conf)
              .Attr<std::string>("grad_mode", "auto")
              .Attr<std::vector<std::string>>("grad_parallel_distribution",
                                              std::vector<std::string>())
              .ScopeSymbolId(scope_symbol_id)
              .Build();
      job_builder->AddOps(node->parallel_desc().parallel_conf(), {parallel_cast_sum_op.op_conf()});
      reduce_sum_op_out = parallel_cast_sum_op.output("out", 0);
    }
    auto broadcast_div_op = user_op::UserOpConfWrapperBuilder(op_name + "-split_softmax_div")
                                .Op("broadcast_div")
                                .Input("x", exp_op.output("y", 0))
                                .Input("y", reduce_sum_op_out)
                                .Output("z")
                                .Build();
    job_builder->AddOps(node->parallel_desc().parallel_conf(), {broadcast_div_op.op_conf()});
    UpdateProbConsumerOpConf(broadcast_div_op.output("z", 0), node, job_builder);

    auto sparse_cross_entropy_ms_op = user_op::UserOpConfWrapperBuilder(op_name)
                                          .Op("sparse_cross_entropy_ms")
                                          .Input("prediction", broadcast_div_op.output("z", 0))
                                          .Input("label", op_label_blob_name)
                                          .Output("out")
                                          .Attr("depth", depth)
                                          .Build();

    job_builder->MutOpsOnlyOnce({sparse_cross_entropy_ms_op.op_conf()});
  });
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("SplitSparseSoftmaxCrossEntropyOpPass", SplitSparseSoftmaxCrossEntropyOpPass);

}  // namespace

}  // namespace oneflow
