#include "oneflow/core/graph/loss_print_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void LossPrintCompTaskNode::ProduceAllRegstsAndBindEdges() {}

void LossPrintCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInEdge()->GetSoleRegst());
}

void LossPrintCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<const Operator> op = chain_node()->SoleOp();
  CHECK(op->IsPrintOp());
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  exec_node->BindBnInOpAndRegst(op->SoleIbn(), GetConsumedRegst("in"));
}

void LossPrintCompTaskNode::FixThrdId() {
  set_thrd_id(IDMgr::Singleton()->AllocatePersistenceThrdId(machine_id()));
}

}  // namespace oneflow
