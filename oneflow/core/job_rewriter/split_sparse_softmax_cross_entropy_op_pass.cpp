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

    user_op::UserOpConfWrapper cur_op(op_conf);
    const std::string op_prediction_blob_name = cur_op.input("prediction", 0);
    const std::string op_label_blob_name = cur_op.input("label", 0);
    const int64_t depth = cur_op.attr<int64_t>("depth");
    std::vector<int32_t> axis_vec(1, 1);

    std::string op_name = node->op().op_name();

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

    const int32_t split_axis =
        node->LogicalBlobDesc4Lbi(node->op().BnInOp2Lbi("prediction_0")).shape().NumAxes() - 1;
    SbpSignature reduce_max_device_stage_sbp_signature;
    (*reduce_max_device_stage_sbp_signature.mutable_bn_in_op2sbp_parallel())["in_0"]
        .mutable_split_parallel()
        ->set_axis(split_axis);
    (*reduce_max_device_stage_sbp_signature.mutable_bn_in_op2sbp_parallel())["out_0"]
        .mutable_split_parallel()
        ->set_axis(split_axis);
    (*job_builder->mutable_job_parallel_view_conf()
          ->mutable_op_name2sbp_signature_conf())[reduce_max_device_stage_op.op_name()] =
        reduce_max_device_stage_sbp_signature;

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

    SbpSignature reduce_max_global_stage_sbp_signature;
    (*reduce_max_global_stage_sbp_signature.mutable_bn_in_op2sbp_parallel())["in_0"]
        .mutable_broadcast_parallel();
    (*reduce_max_global_stage_sbp_signature.mutable_bn_in_op2sbp_parallel())["out_0"]
        .mutable_broadcast_parallel();
    (*job_builder->mutable_job_parallel_view_conf()
          ->mutable_op_name2sbp_signature_conf())[reduce_max_global_stage_op.op_name()] =
        reduce_max_global_stage_sbp_signature;

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

    auto broadcast_div_op = user_op::UserOpConfWrapperBuilder(op_name + "-split_softmax_div")
                                .Op("broadcast_div")
                                .Input("x", exp_op.output("y", 0))
                                .Input("y", reduce_sum_op.output("output_tensor", 0))
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
