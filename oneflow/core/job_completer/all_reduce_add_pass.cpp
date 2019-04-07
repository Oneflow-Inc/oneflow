#include "oneflow/core/job_completer/all_reduce_add_pass.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

namespace {

std::function<const OpNode*(const LogicalBlobId&)> MakeGetterProducerOpNode4Lbi(
    const OpGraph& op_graph) {
  auto lbi2producer_op_node = std::make_shared<HashMap<LogicalBlobId, const OpNode*>>();
  op_graph.ForEachNode([&](OpNode* op_node) {
    for (const std::string& obn : op_node->op().output_bns()) {
      CHECK(lbi2producer_op_node->emplace(op_node->op().BnInOp2Lbi(obn), op_node).second);
    }
  });
  return [lbi2producer_op_node](const LogicalBlobId& lbi) -> const OpNode* {
    return lbi2producer_op_node->at(lbi);
  };
}

void GroupAllReduceLbi(const Job& job,
                       const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi,
                       std::vector<std::vector<LogicalBlobId>>* lbi_groups) {
  TODO();
}

void BuildAllReduceStruct(
    const JobBuilder& job_builder,
    const std::function<const OpNode*(const LogicalBlobId&)>& ProducerOpNode4Lbi,
    const std::vector<LogicalBlobId>& lbi_groups) {
  TODO();
}

}  // namespace

void AllReduceAddPass::Apply(Job* job) const {
  OpGraph op_graph(*job);
  auto ProducerOpNode4Lbi = MakeGetterProducerOpNode4Lbi(op_graph);
  std::vector<std::vector<LogicalBlobId>> lbi_groups;
  GroupAllReduceLbi(*job, ProducerOpNode4Lbi, &lbi_groups);
  JobBuilder job_builder(job);
  for (const auto& lbi_group : lbi_groups) {
    BuildAllReduceStruct(job_builder, ProducerOpNode4Lbi, lbi_group);
  }
}

}  // namespace oneflow
