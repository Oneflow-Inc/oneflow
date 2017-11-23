#include "oneflow/core/graph/loss_accumulate_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void LossAccCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> in_regst = GetConsumedRegst("one");
  std::shared_ptr<RegstDesc> acc_regst = GetProducedRegst("acc");
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = chain_node()->SoleOp();
  std::shared_ptr<const Operator> op = exec_node->op();
  exec_node->BindBnInOpAndRegst(op->SoleIbn(), in_regst);
  acc_regst->AddLbn(op->Lbn4BnInOp(op->SoleObn()));
  exec_node->BindBnInOpAndRegst(op->SoleObn(), acc_regst);
  op->InferBlobDescs(exec_node->GetBlobDesc4BnInOpFunc(), parallel_ctx());
}

}  // namespace oneflow
