#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/graph/optimizer_compute_task_node.h"

namespace oneflow {

void OptimizerCompTaskNode::ConsumeAllRegsts() {
  ForEachInDataEdge([&](TaskEdge* edge) {
    for (const auto& regst : edge->GetRegsts()) { ConsumeRegst("in", regst); }
  });
}

void OptimizerCompTaskNode::ProduceAllRegstsAndBindEdges() { ProduceRegst("tmp", false, 1, 1); }

void OptimizerCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> sole_op = this->logical_node()->SoleOp();
  node->mut_op() = sole_op;
  const std::list<std::shared_ptr<RegstDesc>>& in_regsts = GetConsumedRegst("in");
  for (const auto& ibn : node->op()->input_bns()) {
    node->BindBnWithOneOfTheRegsts(ibn, in_regsts);
  }
  node->AddBnToRegstAndBindIt(&Operator::tmp_bns, GetProducedRegst("tmp"));
  node->InferBlobDescs(parallel_ctx());
}

void OptimizerCompTaskNode::InferProducedDataRegstTimeShape() {
  ForEachProducedDataRegst([](const std::string& name, RegstDesc* regst) {
    regst->mut_data_regst_time_shape()->reset(
        new Shape({GlobalJobDesc().TotalBatchNum(), static_cast<int64_t>(1)}));
  });
}

}  // namespace oneflow
