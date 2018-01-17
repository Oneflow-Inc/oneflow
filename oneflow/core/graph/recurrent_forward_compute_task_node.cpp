#include "oneflow/core/graph/forward_compute_task_node.h"
#include "oneflow/core/graph/recurrent_forward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void RecurrentForwardCompTaskNode::VirtualAddRegstOnRecurrentOutEdge(
    TaskEdge* edge) {
  edge->AddRegst("rec_out", ProduceRegst("rec_out", 1, 1));
}

void RecurrentForwardCompTaskNode::VirtualConsumeInRegst(TaskEdge* edge) {
  std::shared_ptr<const Operator> op = chain_node()->SoleOp();
  std::shared_ptr<RegstDesc> regst = edge->GetSoleRegst();
  const HashSet<std::string>& lbns =
      PredChainNodeOnEdge(edge)->data_output_lbns();
  if (lbns.find(op->Lbn4BnInOp("in")) != lbns.end()) {
    ConsumeRegst("in", regst);
  } else if (lbns.find(op->Lbn4BnInOp("h0")) != lbns.end()) {
    ConsumeRegst("h0", regst);
  } else if (lbns.find(op->Lbn4BnInOp("rec_in")) != lbns.end()) {
    if (parallel_ctx()->policy() == kModelParallel) {
      ConsumeRegst("rec_in", regst);
    }
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
  if (parallel_ctx()->policy() == kModelParallel) {
    exec_node->BindBnInOpAndRegst("rec_in", GetConsumedRegst("rec_in"));
  } else if (parallel_ctx()->policy() == kDataParallel) {
    exec_node->BindBnInOpAndRegst("rec_in", GetProducedRegst("rec_out"));
  } else {
    UNEXPECTED_RUN();
  }
  std::shared_ptr<RegstDesc> h0_regst = GetConsumedRegst("h0");
  if (h0_regst) { exec_node->BindBnInOpAndRegst("h0", h0_regst); }
}

void RecurrentForwardCompTaskNode::BuildOutRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  std::shared_ptr<RegstDesc> rec_out_regst = GetProducedRegst("rec_out");
  CHECK(out_regst && rec_out_regst);
  ExecNode* exec_node = mut_exec_gph().SoleNode();
  const std::string& out_lbn = exec_node->op()->Lbn4BnInOp("out");
  const std::string& rec_out_lbn = exec_node->op()->Lbn4BnInOp("rec_out");
  out_regst->AddLbn(out_lbn);
  rec_out_regst->AddLbn(rec_out_lbn);
  exec_node->BindBnInOpAndRegst("out", out_regst);
  exec_node->BindBnInOpAndRegst("rec_out", rec_out_regst);
}

bool RecurrentForwardCompTaskNode::IsReadyForBuild() {
  std::shared_ptr<RegstDesc> regst = GetConsumedRegst("h0");
  if (GetConsumedRegst("in")->IsLocked() && (!regst || regst->IsLocked())) {
    return true;
  }
  return false;
}

}  // namespace oneflow
