#include "oneflow/core/graph/forward_compute_task_node.h"
#include "oneflow/core/graph/recurrent_backward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

bool RecurrentBackwardCompTaskNode::IsReadyForBuild() {
  auto consumed_regsts_ = consumed_regsts();
  for (auto& pair : consumed_regsts_) {
    if (pair.first == "ht_diff") { continue; }
    if (pair.second.lock()->IsLocked() == false) { return false; }
  }
  return true;
}

void RecurrentBackwardCompTaskNode::VirtualBuildExecGphAndBindOutDiffRegst() {
  std::shared_ptr<const Operator> op = chain_node()->SoleOp();
  CHECK(op->IsRecurrentOp());
  ExecNode* exec_node = mut_exec_gph().NewNode();
  exec_node->mut_op() = op;
  std::shared_ptr<RegstDesc> out_regst = GetConsumedRegst("out");
  std::shared_ptr<RegstDesc> out_diff_regst = GetConsumedRegst("out_diff");
  exec_node->BindBnInOpAndRegst("out", out_regst);
  exec_node->BindBnInOpAndRegst("out_diff", out_diff_regst);
  std::shared_ptr<RegstDesc> ht_regst = GetConsumedRegst("ht");
  std::shared_ptr<RegstDesc> ht_diff_regst = GetConsumedRegst("ht_diff");
  exec_node->BindBnInOpAndRegst("ht", ht_regst);
  exec_node->BindBnInOpAndRegst("ht_diff", ht_diff_regst);
}

void RecurrentBackwardCompTaskNode::VirtualBindInDiffRegst() {
  std::shared_ptr<const Operator> op = chain_node()->SoleOp();
  ExecNode* exec_node = mut_exec_gph().SoleNode();

  exec_node->BindBnInOpAndRegst("in", GetConsumedRegst("in"));

  std::shared_ptr<RegstDesc> in_diff_regst = GetProducedRegst("in_diff");
  in_diff_regst->AddLbn(op->Lbn4BnInOp("in"));
  exec_node->BindBnInOpAndRegst("in_diff", in_diff_regst);

  std::shared_ptr<RegstDesc> ht_1_diff_regst = GetProducedRegst("ht_1_diff");
  ht_1_diff_regst->AddLbn(op->Lbn4BnInOp("ht_1"));
  exec_node->BindBnInOpAndRegst("ht_1_diff", ht_1_diff_regst);

  if (GetConsumedRegst("h0")) {
    exec_node->BindBnInOpAndRegst("h0", GetConsumedRegst("h0"));
    std::shared_ptr<RegstDesc> h0_diff_regst = GetProducedRegst("h0_diff");
    h0_diff_regst->AddLbn(op->Lbn4BnInOp("h0"));
    exec_node->BindBnInOpAndRegst("h0_diff", h0_diff_regst);
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

void RecurrentBackwardCompTaskNode::VirtualProduceRegstOnSelfEdge(
    TaskEdge* edge) {
  edge->AddRegst("ht_1_diff", ProduceRegst("ht_1_diff"));
}

void RecurrentBackwardCompTaskNode::VirtualConsumeDiffRegst(TaskEdge* edge) {
  std::shared_ptr<const Operator> op = chain_node()->SoleOp();
  std::shared_ptr<RegstDesc> regst = edge->GetSoleRegst();
  if (regst->GetBlobDesc(op->Lbn4BnInOp("out_diff"))) {
    ConsumeRegst("out_diff", regst);
  } else if (regst->GetBlobDesc(op->Lbn4BnInOp("ht_diff"))) {
    ConsumeRegst("ht_diff", regst);
  } else {
    UNEXPECTED_RUN();
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

void RecurrentBackwardCompTaskNode::VirtualInferBlobDescInHiddenDiff() {
  std::shared_ptr<RegstDesc> ht_1_diff_regst = GetProducedRegst("ht_1_diff");
  ht_1_diff_regst->CopyBlobDescWithoutAddLbn(GetConsumedRegst("h0").get());
  std::shared_ptr<const Operator> op = chain_node()->SoleOp();
  std::shared_ptr<RegstDesc> ht_diff_regst = GetConsumedRegst("ht_diff");
  if (!ht_diff_regst->IsLocked()) {
    ht_diff_regst->CopyBlobDescFrom(ht_1_diff_regst.get());
  }
  if (std::shared_ptr<RegstDesc> h0_diff_regst = GetConsumedRegst("h0")) {
    h0_diff_regst->CopyBlobDescWithoutAddLbn(GetConsumedRegst("h0").get());
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
