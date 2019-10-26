#include "oneflow/core/job_completer/sparse_softmax_cross_entropy_pass.h"

namespace oneflow {

namespace {

std::string GenReduceMaxMs1OpConf(OperatorConf op_conf, const ParallelConf& parallel_conf,
                                  JobBuilder* job_builder) {
  OperatorConf reduce_max_stage0_op_conf;
  reduce_max_stage0_op_conf.set_name(op_conf.name() + "-softmax_reduce_max_stage0");
  auto* reduce_max_stage0_conf = reduce_max_stage0_op_conf.mutable_device_reduce_max_conf();
  reduce_max_stage0_conf->set_in(op_conf.sparse_softmax_cross_entropy_ms1_conf().prediction());
  reduce_max_stage0_conf->set_out("out");
  reduce_max_stage0_conf->add_axis(1);
  job_builder->AddOps(parallel_conf, {reduce_max_stage0_op_conf});

  OperatorConf reduce_max_stage1_op_conf;
  reduce_max_stage1_op_conf.set_name(op_conf.name() + "-softmax_reduce_max_stage1");
  auto* reduce_max_stage1_conf = reduce_max_stage1_op_conf.mutable_reduce_max_conf();
  reduce_max_stage1_conf->set_in(reduce_max_stage0_op_conf.name() + "/out");
  reduce_max_stage1_conf->set_out("out");
  reduce_max_stage1_conf->add_axis(1);
  reduce_max_stage1_conf->set_keep_dims(true);
  job_builder->AddOps(parallel_conf, {reduce_max_stage1_op_conf});

  SbpSignature reduce_max1_sbp_signature;
  (*reduce_max1_sbp_signature.mutable_bn_in_op2sbp_parallel())["in"].mutable_broadcast_parallel();
  (*reduce_max1_sbp_signature.mutable_bn_in_op2sbp_parallel())["out"].mutable_broadcast_parallel();
  (*job_builder->mutable_sbp_conf()
        ->mutable_op_name2sbp_signature_conf())[reduce_max_stage1_op_conf.name()] =
      reduce_max1_sbp_signature;
  return reduce_max_stage1_op_conf.name() + "/out";
}

std::string GenSoftmaxMs1OpConf(OperatorConf op_conf, std::string reduce_max_lbn,
                                const ParallelConf& parallel_conf, JobBuilder* job_builder) {
  const auto& sparse_softmax_cross_entropy_conf = op_conf.sparse_softmax_cross_entropy_ms1_conf();

  OperatorConf broadcast_sub_op_conf;
  broadcast_sub_op_conf.set_name(op_conf.name() + "-softmax_submax");
  auto* broadcast_sub_conf = broadcast_sub_op_conf.mutable_broadcast_sub_conf();
  broadcast_sub_conf->set_a(sparse_softmax_cross_entropy_conf.prediction());
  broadcast_sub_conf->set_b(reduce_max_lbn);
  broadcast_sub_conf->set_out("out");
  job_builder->AddOps(parallel_conf, {broadcast_sub_op_conf});

  OperatorConf exp_op_conf;
  exp_op_conf.set_name(op_conf.name() + "-softmax_exp");
  auto* exp_conf = exp_op_conf.mutable_exp_conf();
  exp_conf->set_in(broadcast_sub_op_conf.name() + "/out");
  exp_conf->set_out("out");
  job_builder->AddOps(parallel_conf, {exp_op_conf});

  OperatorConf reduce_sum_op_conf;
  reduce_sum_op_conf.set_name(op_conf.name() + "-softmax_reduce_sum");
  auto* reduce_sum_conf = reduce_sum_op_conf.mutable_reduce_sum_conf();
  reduce_sum_conf->set_in(exp_op_conf.name() + "/out");
  reduce_sum_conf->set_out("out");
  reduce_sum_conf->add_axis(1);
  reduce_sum_conf->set_keep_dims(true);
  job_builder->AddOps(parallel_conf, {reduce_sum_op_conf});

  OperatorConf broadcast_div_op_conf;
  broadcast_div_op_conf.set_name(op_conf.name() + "-softmax_div");
  auto* broadcast_div_conf = broadcast_div_op_conf.mutable_broadcast_div_conf();
  broadcast_div_conf->set_a(exp_op_conf.name() + "/out");
  broadcast_div_conf->set_b(reduce_sum_op_conf.name() + "/out");
  broadcast_div_conf->set_out("out");
  job_builder->AddOps(parallel_conf, {broadcast_div_op_conf});

  return broadcast_div_op_conf.name() + "/out";
}

void UpdateSoftmaxProbConsumerOpConf(const std::string& new_prob_lbn, const OpNode* op_node,
                                     JobBuilder* job_builder) {
  const LogicalBlobId& old_prob_lbi = op_node->op().BnInOp2Lbi("prob");
  const std::string& old_prob_lbn = GenLogicalBlobName(old_prob_lbi);
  for (const OpEdge* edge : op_node->out_edges()) {
    OpNode* out_node = edge->dst_node();
    OperatorConf mut_op_conf(out_node->op().op_conf());
    PbMessage* mut_conf = MutableMessageInPbMessage(&mut_op_conf, mut_op_conf.op_type_case());
    if (edge->lbi2ibns().find(old_prob_lbi) != edge->lbi2ibns().end()) {
      const auto& prob_ibns = edge->lbi2ibns().at(old_prob_lbi);
      for (const std::string& ibn : prob_ibns) {
        ReplaceStrValInPbFdOrPbRpf(mut_conf, ibn, old_prob_lbn, new_prob_lbn);
      }
    }
    job_builder->MutOpsOnlyOnce({mut_op_conf});
  }
}

}  // namespace

void SparseSoftmaxCrossEntropyPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  HashMap<std::string, LogicalBlobId> op_name2lbi;
  op_graph.ForEachNode([&](const OpNode* node) {
    if (node->op().op_conf().has_sparse_softmax_cross_entropy_ms1_conf()) {
      const auto& sparse_softmax_cross_entropy_conf =
          node->op().op_conf().sparse_softmax_cross_entropy_ms1_conf();
      std::string reduce_max_lbn = GenReduceMaxMs1OpConf(
          node->op().op_conf(), node->parallel_desc().parallel_conf(), job_builder);
      std::string softmax_prob_lbn = GenSoftmaxMs1OpConf(
          node->op().op_conf(), reduce_max_lbn, node->parallel_desc().parallel_conf(), job_builder);
      UpdateSoftmaxProbConsumerOpConf(softmax_prob_lbn, node, job_builder);
      OperatorConf sparse_cross_entropy_ms1_op_conf(node->op().op_conf());
      auto* sparse_cross_entropy_ms1_conf =
          sparse_cross_entropy_ms1_op_conf.mutable_sparse_cross_entropy_ms1_conf();
      sparse_cross_entropy_ms1_conf->set_prediction(softmax_prob_lbn);
      sparse_cross_entropy_ms1_conf->set_label(sparse_softmax_cross_entropy_conf.label());
      sparse_cross_entropy_ms1_conf->set_depth(sparse_softmax_cross_entropy_conf.depth());
      sparse_cross_entropy_ms1_conf->set_out("out");
      job_builder->MutOpsOnlyOnce({sparse_cross_entropy_ms1_op_conf});
    }
  });
}

}  // namespace oneflow
