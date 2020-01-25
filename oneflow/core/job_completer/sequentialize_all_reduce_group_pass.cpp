#include "oneflow/core/job_completer/op_graph_pass.h"

namespace oneflow {

namespace {

struct AllReduceGroup {
  const OpNode* reduce_identity_node;
  const OpNode* reduce_split_node;
  int64_t order_in_graph;
};

bool IsReduceIdentityNode(const OpNode* node) {
  return node->op().op_conf().has_reduce_identity_conf();
}

bool IsReduceSplitNode(const OpNode* node) { return node->op().op_conf().has_reduce_split_conf(); }

const OpNode* GetSuccReduceSplitNode(const OpNode* node) {
  while (!IsReduceSplitNode(node)) { node = node->SoleOutEdge()->dst_node(); }
  return node;
}

void FindAllReduceGroups(const OpGraph& op_graph, std::vector<AllReduceGroup>* all_reduce_groups) {
  op_graph.ForEachNode([&](const OpNode* node) {
    if (IsReduceIdentityNode(node)) {
      const OpNode* split_node = GetSuccReduceSplitNode(node);
      const int64_t order_in_graph = node->op().op_conf().reduce_identity_conf().order_in_graph();
      CHECK_EQ(split_node->op().op_conf().reduce_split_conf().order_in_graph(), order_in_graph);
      AllReduceGroup all_reduce_group{};
      all_reduce_group.reduce_identity_node = node;
      all_reduce_group.reduce_split_node = split_node;
      all_reduce_group.order_in_graph = order_in_graph;
      all_reduce_groups->emplace_back(all_reduce_group);
    }
  });
}

void ReOrderAllReduceGroups(std::vector<AllReduceGroup>* all_reduce_groups) {
  std::vector<AllReduceGroup> tmp_groups = *all_reduce_groups;
  std::sort(tmp_groups.begin(), tmp_groups.end(),
            [](const AllReduceGroup& lhs, const AllReduceGroup& rhs) {
              return lhs.order_in_graph < rhs.order_in_graph;
            });
  const int32_t group_count = all_reduce_groups->size();
  const int32_t lazy_count = std::round(GlobalJobDesc().all_reduce_lazy_ratio() * group_count);
  CHECK_GE(lazy_count, 0);
  CHECK_LE(lazy_count, group_count);
  std::reverse_copy(tmp_groups.cbegin() + lazy_count, tmp_groups.cend(),
                    all_reduce_groups->begin());
  std::copy(tmp_groups.cbegin(), tmp_groups.cbegin() + lazy_count,
            all_reduce_groups->end() - lazy_count);
}

class SequentializeAllReduceGroupPass final : public OpGraphPass {
 public:
  SequentializeAllReduceGroupPass() = default;
  ~SequentializeAllReduceGroupPass() = default;
  bool IsEnabled() const override {
    return GlobalJobDesc().IsTrain() && !GlobalJobDesc().disable_all_reduce_sequence();
  }
  void Apply(const OpGraph& op_graph, JobBuilder* job_builder) const override;
};

void SequentializeAllReduceGroupPass::Apply(const OpGraph& op_graph,
                                            JobBuilder* job_builder) const {
  std::vector<AllReduceGroup> all_reduce_groups;
  FindAllReduceGroups(op_graph, &all_reduce_groups);
  ReOrderAllReduceGroups(&all_reduce_groups);
  FOR_RANGE(int32_t, i, 1, all_reduce_groups.size()) {
    const std::string& pred_split_op_name =
        all_reduce_groups.at(i - 1).reduce_split_node->op().op_name();
    OperatorConf succ_identity_op_conf =
        all_reduce_groups.at(i).reduce_identity_node->op().op_conf();
    succ_identity_op_conf.add_ctrl_in_op_name(pred_split_op_name);
    job_builder->MutOpsOnlyOnce({succ_identity_op_conf});
  }
}

REGISTER_FUNCTION_PASS("SequentializeAllReduceGroupPass", SequentializeAllReduceGroupPass);

}  // namespace

}  // namespace oneflow
