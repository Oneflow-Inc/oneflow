#include "oneflow/core/graph/forward_compute_task_node.h"
#include "oneflow/core/graph/recurrent_forward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void RecurrentForwardCompTaskNode::VirtualAddRegstForRecurrentOutEdge(
    TaskEdge* edge) {
  edge->AddRegst("ht", ProduceRegst("ht"));
}

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

void RecurrentForwardCompTaskNode::VirtualBindOutRegst() {
  std::shared_ptr<RegstDesc> ht_regst = GetProducedRegst("ht");
  CHECK(ht_regst != NULL);
  ExecNode* exec_node = mut_exec_gph().SoleNode();
  const std::string& ht_lbn = exec_node->op()->Lbn4BnInOp("ht");
  ht_regst->AddLbn(ht_lbn);
  exec_node->BindBnInOpAndRegst("ht", ht_regst);
}

bool RecurrentForwardCompTaskNode::IsReadyForBuild() {
  std::shared_ptr<RegstDesc> regst = GetConsumedRegst("h0");
  if (GetConsumedRegst("in")->IsLocked() && (!regst || regst->IsLocked())) {
    return true;
  }
  return false;
}

void RecurrentForwardCompTaskNode::VirtualFixRegisterNumRange() {
  GetProducedRegst("ht")->set_min_register_num(2);
  GetProducedRegst("ht")->set_max_register_num(2);
}

}  // namespace oneflow
