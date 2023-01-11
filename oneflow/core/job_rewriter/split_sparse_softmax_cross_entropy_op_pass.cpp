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

bool NeedDoPass(const Job& job) {
  return std::any_of(job.net().op().cbegin(), job.net().op().cend(), [&](const OperatorConf& op) {
    return op.has_user_conf() && op.user_conf().op_type_name() == "sparse_softmax_cross_entropy_ms";
  });
}

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
    if (!NeedDoPass(*job)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> SplitSparseSoftmaxCrossEntropyOpPass::Apply(const OpGraph& op_graph,
                                                        JobBuilder* job_builder) const {
  std::vector<std::string> to_del_op_names;
  HashMap<std::string, OperatorConf> consumer_op_name2op_confs;
  op_graph.ForEachNode([&](const OpNode* node) {
    const OperatorConf& op_conf = node->op().op_conf();
    if (!op_conf.has_user_conf()) { return; }
    if (op_conf.user_conf().op_type_name() != "sparse_softmax_cross_entropy_ms") { return; }

    const int64_t scope_symbol_id = node->op().op_conf().scope_symbol_id();
    user_op::UserOpConfWrapper cur_op(op_conf);
    const std::string& op_prediction_blob_name = cur_op.input("prediction", 0);
    const std::string& op_label_blob_name = cur_op.input("label", 0);
    const int32_t split_axis =
        node->LogicalBlobDesc4Lbi(node->op().BnInOp2Lbi("prediction_0")).shape().NumAxes() - 1;
    const std::vector<int32_t> axis_vec(1, split_axis);

    const std::string& op_name = node->op().op_name();
    const auto& prediction_nd_sbp = node->NdSbp4BnInOp("prediction_0");

    NdSbp stat_distribution_for_consumer;

    bool has_split_axis_parallel = false;
    CHECK_EQ(prediction_nd_sbp.sbp_parallel_size(), node->parallel_desc().hierarchy()->NumAxes());
    for (int64_t i = 0; i < node->parallel_desc().hierarchy()->NumAxes(); ++i) {
      const auto& sbp = prediction_nd_sbp.sbp_parallel(i);
      if (sbp.has_split_parallel() && sbp.split_parallel().axis() == split_axis) {
        has_split_axis_parallel = true;
        stat_distribution_for_consumer.add_sbp_parallel()->mutable_broadcast_parallel();
      } else {
        CHECK(!sbp.has_partial_sum_parallel());
        *stat_distribution_for_consumer.add_sbp_parallel() = SbpParallel(sbp);
      }
    }

    if (!has_split_axis_parallel) { return; }
    to_del_op_names.push_back(op_name);

    auto reduce_max_device_stage_op =
        user_op::UserOpConfWrapperBuilder(op_name + "-split_softmax_reduce_max_device_stage")
            .Op("reduce_max_device_stage")
            .Input("in", op_prediction_blob_name)
            .Output("out")
            .Output("mask")
            .Output("count")
            .Attr("axis", axis_vec)
            .ScopeSymbolId(scope_symbol_id)
            .Build();
    job_builder->AddOps(node->parallel_desc().parallel_conf(),
                        {reduce_max_device_stage_op.op_conf()});
    NdSbpSignature reduce_max_device_stage_signature;
    (*reduce_max_device_stage_signature.mutable_bn_in_op2nd_sbp())["in_0"] =
        NdSbp(prediction_nd_sbp);
    (*reduce_max_device_stage_signature.mutable_bn_in_op2nd_sbp())["out_0"] =
        NdSbp(prediction_nd_sbp);
    (*reduce_max_device_stage_signature.mutable_bn_in_op2nd_sbp())["mask_0"] =
        NdSbp(prediction_nd_sbp);
    (*reduce_max_device_stage_signature.mutable_bn_in_op2nd_sbp())["count_0"] =
        NdSbp(prediction_nd_sbp);
    job_builder->AddNdSbpSignature4OpName(reduce_max_device_stage_op.op_name(),
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
            .ScopeSymbolId(scope_symbol_id)
            .Build();
    job_builder->AddOps(node->parallel_desc().parallel_conf(),
                        {reduce_max_global_stage_op.op_conf()});
    NdSbpSignature reduce_max_global_stage_signature;
    (*reduce_max_global_stage_signature.mutable_bn_in_op2nd_sbp())["in_0"] =
        stat_distribution_for_consumer;
    (*reduce_max_global_stage_signature.mutable_bn_in_op2nd_sbp())["device_count_0"] =
        stat_distribution_for_consumer;
    (*reduce_max_global_stage_signature.mutable_bn_in_op2nd_sbp())["out_0"] =
        stat_distribution_for_consumer;
    job_builder->AddNdSbpSignature4OpName(reduce_max_global_stage_op.op_name(),
                                          reduce_max_global_stage_signature);

    auto broadcast_sub_max_op =
        user_op::UserOpConfWrapperBuilder(op_name + "-split_softmax_sub_max")
            .Op("broadcast_sub")
            .Input("x", op_prediction_blob_name)
            .Input("y", reduce_max_global_stage_op.output("out", 0))
            .Output("z")
            .ScopeSymbolId(scope_symbol_id)
            .Build();
    job_builder->AddOps(node->parallel_desc().parallel_conf(), {broadcast_sub_max_op.op_conf()});

    auto exp_op = user_op::UserOpConfWrapperBuilder(op_name + "-split_softmax_exp")
                      .Op("exp")
                      .Input("x", broadcast_sub_max_op.output("z", 0))
                      .Output("y")
                      .ScopeSymbolId(scope_symbol_id)
                      .Build();
    job_builder->AddOps(node->parallel_desc().parallel_conf(), {exp_op.op_conf()});

    auto reduce_sum_op = user_op::UserOpConfWrapperBuilder(op_name + "-split_softmax_reduce_sum")
                             .Op("reduce_sum")
                             .Input("input_tensor", exp_op.output("y", 0))
                             .Output("output_tensor")
                             .Attr("axis", axis_vec)
                             .Attr("keepdims", true)
                             .ScopeSymbolId(scope_symbol_id)
                             .Build();
    job_builder->AddOps(node->parallel_desc().parallel_conf(), {reduce_sum_op.op_conf()});

    std::string reduce_sum_op_out;
    if (node->parallel_desc().hierarchy()->NumAxes() > 1) {
      std::vector<std::string> nd_sbp_conf;
      for (const auto& sbp_parallel : stat_distribution_for_consumer.sbp_parallel()) {
        nd_sbp_conf.emplace_back(SbpParallelToString(sbp_parallel));
      }
      auto parallel_cast_sum_op =
          user_op::UserOpConfWrapperBuilder(op_name + "-split_softmax_reduce_sum_cast_P2B")
              .Op("hierarchical_parallel_cast")
              .Input("in", reduce_sum_op.output("output_tensor", 0))
              .Output("out")
              .Attr<std::vector<std::string>>("nd_sbp", nd_sbp_conf)
              .Attr<std::string>("grad_mode", "auto")
              .Attr<std::vector<std::string>>("grad_nd_sbp", std::vector<std::string>())
              .ScopeSymbolId(scope_symbol_id)
              .Build();
      job_builder->AddOps(node->parallel_desc().parallel_conf(), {parallel_cast_sum_op.op_conf()});
      reduce_sum_op_out = parallel_cast_sum_op.output("out", 0);
    } else {
      reduce_sum_op_out = reduce_sum_op.output("output_tensor", 0);
    }

    auto broadcast_div_op = user_op::UserOpConfWrapperBuilder(op_name + "-split_softmax_div")
                                .Op("broadcast_div")
                                .Input("x", exp_op.output("y", 0))
                                .Input("y", reduce_sum_op_out)
                                .Output("z")
                                .ScopeSymbolId(scope_symbol_id)
                                .Build();
    job_builder->AddOps(node->parallel_desc().parallel_conf(), {broadcast_div_op.op_conf()});

    auto log_op = user_op::UserOpConfWrapperBuilder(op_name + "-log")
                      .Op("log")
                      .Input("x", reduce_sum_op_out)
                      .Output("y")
                      .ScopeSymbolId(scope_symbol_id)
                      .Build();
    job_builder->AddOps(node->parallel_desc().parallel_conf(), {log_op.op_conf()});

    auto broadcast_sub_op = user_op::UserOpConfWrapperBuilder(op_name + "-broadcast_add")
                                .Op("broadcast_sub")
                                .Input("x", broadcast_sub_max_op.output("z", 0))
                                .Input("y", log_op.output("y", 0))
                                .Output("z")
                                .ScopeSymbolId(scope_symbol_id)
                                .Build();
    job_builder->AddOps(node->parallel_desc().parallel_conf(), {broadcast_sub_op.op_conf()});

    auto nll_op = user_op::UserOpConfWrapperBuilder(op_name + "-nll")
                      .Op("nll")
                      .Input("input", broadcast_sub_op.output("z", 0))
                      .Input("target", op_label_blob_name)
                      .Output("output")
                      .Output("out_weight")
                      .Attr<int64_t>("ignore_index", -100)
                      .ScopeSymbolId(scope_symbol_id)
                      .Build();
    job_builder->AddOps(node->parallel_desc().parallel_conf(), {nll_op.op_conf()});

    const std::string& prob_lbn = cur_op.output("prob", 0);
    const std::string& out_lbn = cur_op.output("out", 0);
    const std::string& new_prob_lbn = broadcast_div_op.output("z", 0);
    const std::string& new_out_lbn = nll_op.output("output", 0);

    for (const OpEdge* out_edge : node->out_edges()) {
      const OpNode* consumer = out_edge->dst_node();
      const std::string& consumer_op_name = consumer->op().op_name();
      if (consumer_op_name2op_confs.find(consumer_op_name) == consumer_op_name2op_confs.end()) {
        consumer_op_name2op_confs[consumer_op_name] = consumer->op().op_conf();
      }
      OperatorConf& consumer_op_conf = consumer_op_name2op_confs[consumer_op_name];
      for (const std::string& ibn : consumer->op().input_bns()) {
        const std::string& input_lbn = GenLogicalBlobName(consumer->op().BnInOp2Lbi(ibn));
        if (input_lbn == prob_lbn) {
          const auto& old_lbn =
              ReplaceInputLbnInOpCustomizedConf(&consumer_op_conf, ibn, new_prob_lbn);
          CHECK_EQ(old_lbn, prob_lbn);
        } else if (input_lbn == out_lbn) {
          const auto& old_lbn =
              ReplaceInputLbnInOpCustomizedConf(&consumer_op_conf, ibn, new_out_lbn);
          CHECK_EQ(old_lbn, out_lbn);
        } else {
          // does not care
        }
      }
    }
  });
  for (const auto& pair : consumer_op_name2op_confs) { job_builder->MutOpsOnlyOnce({pair.second}); }
  job_builder->DelOps(to_del_op_names);
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("SplitSparseSoftmaxCrossEntropyOpPass", SplitSparseSoftmaxCrossEntropyOpPass);

}  // namespace

}  // namespace oneflow
