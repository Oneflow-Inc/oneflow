#include "oneflow/core/graph/loss_accumulate_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void LossAccCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> one_regst = GetConsumedRegst("one");
  std::shared_ptr<RegstDesc> acc_regst = GetProducedRegst("acc");
  acc_regst->CopyBlobDescFrom(one_regst.get());
  std::shared_ptr<const Operator> op = chain_node()->SoleOp();
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  exec_node->BindBnInOpAndRegst(op->SoleIbn(), one_regst);
  exec_node->BindBnInOpAndRegst(op->SoleObn(), acc_regst);
}

}  // namespace oneflow
