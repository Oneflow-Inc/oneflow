#include "oneflow/core/graph/poor_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void PoorCompTaskNode::ProduceAllRegstsAndBindEdges() {
  this->SoleOutEdge()->AddRegst("out", ProduceRegst("out", false, 1, 1));
}

void PoorCompTaskNode::ConsumeAllRegsts() { ConsumeRegst("in", SoleInEdge()->GetSoleRegst()); }

void PoorCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> sole_op = this->logical_node()->SoleOp();
  node->mut_op() = sole_op;
  node->BindBnWithRegst(sole_op->SoleIbn(), GetSoleConsumedRegst("in"));
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  out_regst->AddLbi(sole_op->BnInOp2Lbi(sole_op->SoleObn()));
  node->BindBnWithRegst(sole_op->SoleObn(), out_regst);
  node->InferBlobDescs(parallel_ctx());
}

}  // namespace oneflow

