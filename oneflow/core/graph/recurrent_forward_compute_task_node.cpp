#include "oneflow/core/graph/forward_compute_task_node.h"
#include "oneflow/core/graph/recurrent_forward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void RecurrentForwardCompTaskNode::VirtualConsumeInRegst(TaskEdge* edge) {
  std::shared_ptr<const Operator> op = chain_node()->SoleOp();
  std::shared_ptr<RegstDesc> regst = edge->GetSoleRegst();
  if (regst->GetBlobDesc(op->Lbn4BnInOp("in"))) {
    ConsumeRegst("in", regst);
  } else if (regst->GetBlobDesc(op->Lbn4BnInOp("h0"))) {
    ConsumeRegst("h0", regst);
  } else if (regst->GetBlobDesc(op->Lbn4BnInOp("ht_1"))) {
    ConsumeRegst("ht_1", regst);
  } else {
    UNEXPECTED_RUN();
  }
}

void RecurrentForwardCompTaskNode::BuildExecGphStructAndBindInRegst() {
  std::shared_ptr<const Operator> op = chain_node()->SoleOp();
  CHECK(op->IsRecurrentOp());
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  exec_node->BindBnInOpAndRegst("in", GetConsumedRegst("in"));
  exec_node->BindBnInOpAndRegst("ht_1", GetConsumedRegst("ht_1"));
  std::shared_ptr<RegstDesc> h0_regst = GetConsumedRegst("h0");
  if (h0_regst) { exec_node->BindBnInOpAndRegst("h0", h0_regst); }
}

void RecurrentForwardCompTaskNode::BuildOutRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  std::shared_ptr<RegstDesc> ht_regst = GetProducedRegst("ht");
  CHECK(ht_regst != NULL);
  ExecNode* exec_node = mut_exec_gph().SoleNode();
  const std::string& lbn = exec_node->op()->Lbn4BnInOp("out");
  out_regst->AddLbn(lbn);
  ht_regst->AddLbn(lbn);
  exec_node->BindBnInOpAndRegst("out", out_regst);
  exec_node->BindBnInOpAndRegst("ht", ht_regst);
}

bool RecurrentForwardCompTaskNode::IsReadyForBuild() {
  std::shared_ptr<RegstDesc> regst = GetConsumedRegst("h0");
  if (GetConsumedRegst("in")->IsLocked() && (!regst || regst->IsLocked())) {
    return true;
  }
  return false;
}

}  // namespace oneflow
