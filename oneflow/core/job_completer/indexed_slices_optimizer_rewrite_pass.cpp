#include "oneflow/core/job_completer/op_graph_pass.h"

namespace oneflow {

class IndexedSlicesOptimizerRewritePass final : public OpGraphPass {
 public:
  IndexedSlicesOptimizerRewritePass() = default;
  ~IndexedSlicesOptimizerRewritePass() override = default;
  bool IsEnabled() const override {
    return GlobalJobDesc().job_conf().has_indexed_slices_optimizer_conf()
           && GlobalJobDesc().job_conf().indexed_slices_optimizer_conf().enable();
  }
  void Apply(const OpGraph& op_graph, JobBuilder* job_builder) const override;
};

void IndexedSlicesOptimizerRewritePass::Apply(const OpGraph& op_graph,
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
    std::function<void(OperatorConf * new_optimizer_op_conf, const std::string& indices,
                       const std::string& values)>
        BuildOptimizer;
    if (src_op_conf.has_gather_ms0_grad_conf()) {
      const GatherMs0GradOpConf& gather_ms0_grad_conf = src_op_conf.gather_ms0_grad_conf();
      indices_lbn = gather_ms0_grad_conf.indices();
      values_lbn = gather_ms0_grad_conf.out_diff();
    } else if (src_op_conf.has_unsorted_segment_sum_conf()) {
      const UnsortedSegmentSumOpConf& unsorted_segment_sum_conf =
          src_op_conf.unsorted_segment_sum_conf();
      if (unsorted_segment_sum_conf.axis() == 0) {
        indices_lbn = unsorted_segment_sum_conf.segment_ids();
        values_lbn = unsorted_segment_sum_conf.data();
      }
    } else {
      return;
    }
    std::vector<const OpNode*> op_nodes_to_remove;
    std::vector<const OpNode*> op_nodes_apply_to_diff;
    const OpNode* dst_node = src_node->SoleOutEdge()->dst_node();
    do {
      if (dst_node->op().output_bns().empty()) { break; }
      const OperatorConf& dst_op_conf = dst_node->op().op_conf();
      if (dst_op_conf.has_parallel_cast_conf()) {
        if (dst_node->out_edges().size() != 1) { return; }
        op_nodes_to_remove.push_back(dst_node);
        dst_node = dst_node->SoleOutEdge()->dst_node();
        continue;
      } else if (dst_op_conf.has_scalar_mul_conf()) {
        if (dst_node->out_edges().size() != 1) { return; }
        op_nodes_apply_to_diff.push_back(dst_node);
        dst_node = dst_node->SoleOutEdge()->dst_node();
        continue;
      } else {
        return;
      }
    } while (true);
    const OperatorConf& dst_op_conf = dst_node->op().op_conf();
    if (dst_op_conf.has_naive_model_update_conf()) {
      const NaiveModelUpdateOpConf& old_optimizer_conf = dst_op_conf.naive_model_update_conf();
      const LogicalBlobId& model_lbi = dst_node->op().BnInOp2Lbi("model");
      model_op_name = model_lbi.op_name();
      BuildOptimizer = [&](OperatorConf* new_optimizer_op_conf, const std::string& indices,
                           const std::string& values) {
        IndexedSlicesNaiveModelUpdateOpConf* new_optimizer_conf =
            new_optimizer_op_conf->mutable_indexed_slices_naive_model_update_conf();
        new_optimizer_conf->set_model_diff_indices(indices);
        new_optimizer_conf->set_model_diff_values(values);
        new_optimizer_conf->set_model(old_optimizer_conf.model());
        new_optimizer_conf->set_learning_rate(old_optimizer_conf.learning_rate());
      };
    } else if (dst_op_conf.has_momentum_model_update_conf()) {
      const MomentumModelUpdateOpConf& old_optimizer_conf =
          dst_op_conf.momentum_model_update_conf();
      const LogicalBlobId& model_lbi = dst_node->op().BnInOp2Lbi("model");
      model_op_name = model_lbi.op_name();
      BuildOptimizer = [&](OperatorConf* new_optimizer_op_conf, const std::string& indices,
                           const std::string& values) {
        IndexedSlicesMomentumModelUpdateOpConf* new_optimizer_conf =
            new_optimizer_op_conf->mutable_indexed_slices_momentum_model_update_conf();
        new_optimizer_conf->set_momentum(old_optimizer_conf.momentum());
        new_optimizer_conf->set_model_diff_indices(indices);
        new_optimizer_conf->set_model_diff_values(values);
        new_optimizer_conf->set_model(old_optimizer_conf.model());
        new_optimizer_conf->set_train_step(old_optimizer_conf.train_step());
        new_optimizer_conf->set_learning_rate(old_optimizer_conf.learning_rate());
        new_optimizer_conf->set_beta(old_optimizer_conf.user_conf().momentum_conf().beta());
      };
    } else if (dst_op_conf.has_lazy_adam_model_update_conf()) {
      const LazyAdamModelUpdateOpConf& old_optimizer_conf =
          dst_op_conf.lazy_adam_model_update_conf();
      const LogicalBlobId& model_lbi = dst_node->op().BnInOp2Lbi("model");
      model_op_name = model_lbi.op_name();
      BuildOptimizer = [&](OperatorConf* new_optimizer_op_conf, const std::string& indices,
                           const std::string& values) {
        IndexedSlicesLazyAdamModelUpdateOpConf* new_optimizer_conf =
            new_optimizer_op_conf->mutable_indexed_slices_lazy_adam_model_update_conf();
        new_optimizer_conf->set_m(old_optimizer_conf.m());
        new_optimizer_conf->set_v(old_optimizer_conf.v());
        new_optimizer_conf->set_model_diff_indices(indices);
        new_optimizer_conf->set_model_diff_values(values);
        new_optimizer_conf->set_model(old_optimizer_conf.model());
        new_optimizer_conf->set_train_step(old_optimizer_conf.train_step());
        new_optimizer_conf->set_learning_rate(old_optimizer_conf.learning_rate());
        new_optimizer_conf->set_beta1(old_optimizer_conf.user_conf().lazy_adam_conf().beta1());
        new_optimizer_conf->set_beta2(old_optimizer_conf.user_conf().lazy_adam_conf().beta2());
        new_optimizer_conf->set_epsilon(old_optimizer_conf.user_conf().lazy_adam_conf().epsilon());
      };
    } else {
      return;
    }
    if (!BuildOptimizer) { return; }
    CHECK(!model_op_name.empty());
    CHECK(!indices_lbn.empty());
    CHECK(!values_lbn.empty());
    if (include_op_name_set.find(model_op_name) == include_op_name_set.end()) { return; }
    for (const OpNode* node : op_nodes_to_remove) { job_builder->DelOps({node->op().op_conf()}); }
    for (const OpNode* node : op_nodes_apply_to_diff) {
      OperatorConf new_conf = node->op().op_conf();
      if (new_conf.has_scalar_mul_conf()) {
        new_conf.mutable_scalar_mul_conf()->set_in(values_lbn);
        values_lbn = GenLogicalBlobName(new_conf.name(), new_conf.scalar_mul_conf().out());
        job_builder->MutOpsOnlyOnce({new_conf});
      } else {
        UNIMPLEMENTED();
      }
    }
    OperatorConf new_optimizer_op_conf{};
    new_optimizer_op_conf.set_name("System-Optimizer-IndexedSlices-" + model_op_name);
    BuildOptimizer(&new_optimizer_op_conf, indices_lbn, values_lbn);
    job_builder->DelOps({src_op_conf, dst_op_conf});
    job_builder->AddOps(dst_node->parallel_desc().parallel_conf(), {new_optimizer_op_conf});
  });
}

REGISTER_FUNCTION_PASS("IndexedSlicesOptimizerRewritePass", IndexedSlicesOptimizerRewritePass);

}  // namespace oneflow
