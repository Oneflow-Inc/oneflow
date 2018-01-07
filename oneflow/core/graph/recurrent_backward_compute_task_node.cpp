#include "oneflow/core/graph/forward_compute_task_node.h"
#include "oneflow/core/graph/recurrent_backward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

bool RecurrentBackwardCompTaskNode::IsReadyForBuild() {
  auto consumed_regsts_ = consumed_regsts();
  for (auto& pair : consumed_regsts_) {
    if (pair.first == "ht_1_diff") { continue; }
    if (pair.second.lock()->IsLocked() == false) { return false; }
  }
  return true;
}

void RecurrentBackwardCompTaskNode::VirtualBuildExecGphAndBindOutDiffRegst() {
  std::shared_ptr<const Operator> op = chain_node()->SoleOp();
  CHECK(op->IsRecurrentOp());
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  exec_node->BindBnInOpAndRegst("out", GetConsumedRegst("out"));
  exec_node->BindBnInOpAndRegst("out_diff", GetConsumedRegst("out_diff"));
}

void RecurrentBackwardCompTaskNode::VirtualBuildInDiffRegst() {
  ExecNode* exec_node = mut_exec_gph().SoleNode();
  exec_node->BindBnInOpAndRegst("in", GetConsumedRegst("in"));
  exec_node->BindBnInOpAndRegst("in_diff", GetProducedRegst("in_diff"));
  if (GetConsumedRegst("h0")) {
    exec_node->BindBnInOpAndRegst("h0", GetConsumedRegst("h0"));
    exec_node->BindBnInOpAndRegst("h0_diff", GetProducedRegst("h0_diff"));
  }
}

void RecurrentBackwardCompTaskNode::VirtualProduceInDiffAndBindEdge(
    TaskEdge* edge) {
  if (CanBindInDiffWhenRecurrent(edge)) {
    edge->AddRegst("in_diff", ProduceRegst("in_diff"));
  } else {
    edge->AddRegst("h0_diff", ProduceRegst("h0_diff"));
  }
}

void RecurrentBackwardCompTaskNode::VirtualInferBlobDescInHiddenDiff() {
  auto ht_1_diff_regst = GetProducedRegst("ht_1_diff");
  ht_1_diff_regst->CopyBlobDescWithoutAddLbn(GetConsumedRegst("out").get());
  if (std::shared_ptr<RegstDesc> h0_diff_regst = GetConsumedRegst("h0")) {
    h0_diff_regst->CopyBlobDescWithoutAddLbn(GetConsumedRegst("h0").get());
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

bool RecurrentBackwardCompTaskNode::CanBindInDiffWhenRecurrent(TaskEdge* edge) {
  TaskNode* node = edge->dst_node();
  while (node->GetTaskType() == kBoxing) {
    TaskEdge* edge = *(node->out_edges().begin());
    node = edge->dst_node();
  }
  BackwardCompTaskNode* succ_bw_node = static_cast<BackwardCompTaskNode*>(node);
  ForwardCompTaskNode* pred_fw_node =
      static_cast<ForwardCompTaskNode*>(succ_bw_node->GetRelatedFwTaskNode());
  std::string in_lbn = chain_node()->SoleOp()->Lbn4BnInOp("in");
  for (std::string lbn : pred_fw_node->chain_node()->data_output_lbns()) {
    if (lbn == in_lbn) { return true; }
  }
  return false;
}

}  // namespace oneflow
