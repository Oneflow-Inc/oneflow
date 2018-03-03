#include "oneflow/core/graph/chain_node.h"
#include "oneflow/core/graph/loss_accumulate_compute_task_node.h"

namespace oneflow {

void AccCompTaskNode::ProduceAllRegstsAndBindEdges() {
  auto acc_regst = ProduceRegst("acc");
  SoleOutEdge()->AddRegst("acc", acc_regst);
}

void AccCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("one", SoleInEdge()->GetSoleRegst());
}

void AccCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> one_regst = GetConsumedRegst("one");
  std::shared_ptr<RegstDesc> acc_regst = GetProducedRegst("acc");
  acc_regst->CopyBlobDescFrom(one_regst.get());
  for (std::shared_ptr<const Operator> op : chain_node()->op_vec()) {
    ExecNode* exec_node = mut_exec_gph().NewNode();
    exec_node->mut_op() = op;
    exec_node->BindBnInOpAndRegst(op->SoleIbn(), one_regst);
    acc_regst->AddLbn(op->Lbn4BnInOp(op->SoleObn()));
    exec_node->BindBnInOpAndRegst(op->SoleObn(), acc_regst);
  }
}

}  // namespace oneflow
