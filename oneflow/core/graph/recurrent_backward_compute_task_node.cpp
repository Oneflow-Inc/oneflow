#include "oneflow/core/graph/recurrent_backward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

bool RecurrentBackwardCompTaskNode::IsReadyForBuild() {
  auto consumed_regsts_ = GetConsumedRegsts();
  for (auto& pair : consumed_regsts_) {
    if (pair.first == "ht_1_diff") { continue; }
    if (pair.second.lock()->IsLocked() == false) { return false; }
  }
  return true;
}

void RecurrentBackwardCompTaskNode::BuildExecGphAndBindOutDiffRegst() {
  std::shared_ptr<const Operator> op = chain_node()->SoleOp();
  CHECK(op->IsRecurrentOp());
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  exec_node->BindBnInOpAndRegst("out", GetConsumedRegst("out"));
  exec_node->BindBnInOpAndRegst("out_diff", GetConsumedRegst("out_diff"));
}

void RecurrentBackwardCompTaskNode::BuildInDiffRegst() {
  ExecNode* exec_node = mut_exec_gph().SoleNode();
  exec_node->BindBnInOpAndRegst("in", GetConsumedRegst("in"));
  exec_node->BindBnInOpAndRegst("in_diff", GetProducedRegst("in_diff"));
  if (GetConsumedRegst("h0")) {
    exec_node->BindBnInOpAndRegst("h0", GetConsumedRegst("h0"));
    exec_node->BindBnInOpAndRegst("h0_diff", GetProducedRegst("h0_diff"));
  }
}

void RecurrentBackwardCompTaskNode::VirtualConsumeInRegst() {
  CompTaskNode* fw_node = static_cast<CompTaskNode*>(GetRelatedFwTaskNode());
  std::shared_ptr<const Operator> op = fw_node->chain_node()->SoleOp();
  for (TaskEdge* edge : fw_node->in_edges()) {
    TaskNode* pred_fw_node = edge->src_node();
    if (pred_fw_node->GetTaskType() == TaskType::kMdUpdt) { continue; }
    std::shared_ptr<RegstDesc> regst = edge->GetSoleRegst();
    if (regst->GetBlobDesc(op->Lbn4BnInOp("in"))) {
      ConsumeRegst("in", regst);
    } else if (regst->GetBlobDesc(op->Lbn4BnInOp("h0"))) {
      ConsumeRegst("h0", regst);
    } else {
      UNEXPECTED_RUN();
    }
  }
}

}  // namespace oneflow
