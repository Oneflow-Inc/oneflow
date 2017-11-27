#include "oneflow/core/graph/loss_record_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void LossRecordCompTaskNode::ProduceAllRegstsAndBindEdges() {}

void LossRecordCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInEdge()->GetSoleRegst());
}

void LossRecordCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<const Operator> op = chain_node()->SoleOp();
  CHECK(op->IsRecordOp());
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  exec_node->BindBnInOpAndRegst(op->SoleIbn(), GetConsumedRegst("in"));
}

void LossRecordCompTaskNode::FixThrdId() {
  set_thrd_id(IDMgr::Singleton()->AllocatePersistenceThrdId(machine_id()));
}

}  // namespace oneflow
