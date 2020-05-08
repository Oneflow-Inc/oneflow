#include "oneflow/core/job_completer/op_graph_pass.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

void UpdateSoftmaxProbConsumerOpConf(const std::string& new_prob_lbn, const OpNode* op_node,
                                     JobBuilder* job_builder) {
  for (const OpEdge* edge : op_node->out_edges()) {
    OpNode* out_node = edge->dst_node();
    auto* op_conf = job_builder->MutableOpConf4OpName(out_node->op().op_name());
    if (op_conf->user_conf().op_type_name() == "sparse_softmax_cross_entropy_ms_grad") {
      auto it = op_conf->user_conf().input().find("prob");
      CHECK(it != op_conf->user_conf().input().end());
      (*(op_conf->mutable_user_conf()->mutable_input()))["prob"].set_s(0, new_prob_lbn);
    }
  }
}

class SplitSparseSoftmaxCrossEntropyOpPass final : public OpGraphPass {
 public:
  SplitSparseSoftmaxCrossEntropyOpPass() = default;
  ~SplitSparseSoftmaxCrossEntropyOpPass() override = default;
  bool IsEnabled() const override {
    // if has sparse softmax cross entropy ms op
    return true;
  }
  void Apply(const OpGraph& op_graph, JobBuilder* job_builder) const override;
};

void SplitSparseSoftmaxCrossEntropyOpPass::Apply(const OpGraph& op_graph,
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

    user_op::UserOpConfWrapperBuilder reduce_max_device_stage_builder(
        op_name + "-split_softmax_reduce_max_device_stage");
    user_op::UserOpConfWrapper reduce_max_device_stage_op =
        reduce_max_device_stage_builder
            .Op("reduce_max")  // max")
            .Input("input_tensor", op_prediction_blob_name)
            .Output("output_tensor")
            .Attr("axis", axis_vec)
            .Attr("keepdims", true)
            .Build();
    job_builder->AddOps(node->parallel_desc().parallel_conf(),
                        {reduce_max_device_stage_op.op_conf()});

    user_op::UserOpConfWrapperBuilder reduce_max_global_stage_builder(
        op_name + "-split_softmax_reduce_max_global_stage");
    user_op::UserOpConfWrapper reduce_max_global_stage_op =
        reduce_max_global_stage_builder
            .Op("reduce_max")  // max")
            .Input("input_tensor", reduce_max_device_stage_op.output("output_tensor", 0))
            .Output("output_tensor")
            .Attr("axis", axis_vec)
            .Attr("keepdims", true)
            .Build();
    job_builder->AddOps(node->parallel_desc().parallel_conf(),
                        {reduce_max_global_stage_op.op_conf()});

    SbpSignature reduce_max_global_stage_sbp_signature;
    (*reduce_max_global_stage_sbp_signature.mutable_bn_in_op2sbp_parallel())["input_tensor_0"]
        .mutable_broadcast_parallel();
    (*reduce_max_global_stage_sbp_signature.mutable_bn_in_op2sbp_parallel())["output_tensor_0"]
        .mutable_broadcast_parallel();
    (*job_builder->mutable_sbp_conf()
          ->mutable_op_name2sbp_signature_conf())[reduce_max_global_stage_op.op_name()] =
        reduce_max_global_stage_sbp_signature;

    // user_op::UserOpConfWrapperBuilder sub_max_builder(op_name + "-split_softmax_sub_max");
    // user_op::UserOpConfWrapper broadcast_sub_max_op =
    //    sub_max_builder.Op("broadcast_sub")
    //        .Input("a", op_prediction_blob_name)
    //        .Input("b", reduce_max_global_stage_op.output("output_tensor", 0))
    //        .Output("out")
    //        .Build();
    // job_builder->AddOps(node->parallel_desc().parallel_conf(), {broadcast_sub_max_op.op_conf()});

    OperatorConf broadcast_sub_op_conf;
    broadcast_sub_op_conf.set_name(node->op().op_name() + "-softmax_submax");
    auto* broadcast_sub_conf = broadcast_sub_op_conf.mutable_broadcast_sub_conf();
    broadcast_sub_conf->set_a(op_prediction_blob_name);
    broadcast_sub_conf->set_b(reduce_max_global_stage_op.output("output_tensor", 0));
    broadcast_sub_conf->set_out("out");
    job_builder->AddOps(node->parallel_desc().parallel_conf(), {broadcast_sub_op_conf});

    user_op::UserOpConfWrapperBuilder exp_builder(op_name + "-split_softmax_exp");
    // user_op::UserOpConfWrapper exp_op =
    //    exp_builder.Op("exp").Input("x", broadcast_sub_max_op.output("out",
    //    0)).Output("y").Build();
    user_op::UserOpConfWrapper exp_op = exp_builder.Op("leaky_relu")
                                            .Input("x", broadcast_sub_op_conf.name() + "/out")
                                            .Output("y")
                                            .Attr("alpha", float(0.1))
                                            .Build();
    job_builder->AddOps(node->parallel_desc().parallel_conf(), {exp_op.op_conf()});

    user_op::UserOpConfWrapperBuilder reduce_sum_builder(op_name + "-split_softmax_reduce_sum");
    user_op::UserOpConfWrapper reduce_sum_op = reduce_sum_builder.Op("reduce_sum")
                                                   .Input("input_tensor", exp_op.output("y", 0))
                                                   .Output("output_tensor")
                                                   .Attr("axis", axis_vec)
                                                   .Attr("keepdims", true)
                                                   .Build();
    job_builder->AddOps(node->parallel_desc().parallel_conf(), {reduce_sum_op.op_conf()});

    // user_op::UserOpConfWrapperBuilder div_builder(op_name + "-split_softmax_div");
    // user_op::UserOpConfWrapper broadcast_div_op = div_builder.Op("broadcast_div")
    //                                                  .Input("a", exp_op.output("y", 0))
    //                                                  .Input("b", reduce_sum_op.output("y", 0))
    //                                                  .Output("out")
    //                                                  .Build();
    // job_builder->AddOps(node->parallel_desc().parallel_conf(), {broadcast_div_op.op_conf()});
    // UpdateSoftmaxProbConsumerOpConf(broadcast_div_op.output("out", 0), node, job_builder);

    OperatorConf broadcast_div_op_conf;
    broadcast_div_op_conf.set_name(node->op().op_name() + "-softmax_div");
    auto* broadcast_div_conf = broadcast_div_op_conf.mutable_broadcast_div_conf();
    broadcast_div_conf->set_a(exp_op.output("y", 0));
    broadcast_div_conf->set_b(reduce_sum_op.output("output_tensor", 0));
    broadcast_div_conf->set_out("out");
    job_builder->AddOps(node->parallel_desc().parallel_conf(), {broadcast_div_op_conf});

    user_op::UserOpConfWrapperBuilder sparse_cross_entropy_builder(op_name);
    user_op::UserOpConfWrapper sparse_cross_entropy_ms_op =
        sparse_cross_entropy_builder.Op("sparse_cross_entropy_ms")
            .Input("prediction",
                   broadcast_div_op_conf.name() + "/out")  // broadcast_div_op.output("out", 0))
            .Input("label", op_label_blob_name)
            .Output("out")
            .Attr("depth", depth)
            .Build();

    std::string prob_lbn = broadcast_div_op_conf.name() + "/out";
    UpdateSoftmaxProbConsumerOpConf(prob_lbn, node, job_builder);

    job_builder->MutOpsOnlyOnce({sparse_cross_entropy_ms_op.op_conf()});
  });
}

REGISTER_FUNCTION_PASS("SplitSparseSoftmaxCrossEntropyOpPass",
                       SplitSparseSoftmaxCrossEntropyOpPass);

}  // namespace

}  // namespace oneflow
