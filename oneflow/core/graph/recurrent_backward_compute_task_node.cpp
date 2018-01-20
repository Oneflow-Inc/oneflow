#include "oneflow/core/graph/forward_compute_task_node.h"
#include "oneflow/core/graph/recurrent_backward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

bool RecurrentBackwardCompTaskNode::IsReadyForBuild() {
  auto consumed_regsts_ = consumed_regsts();
  for (auto& pair : consumed_regsts_) {
    if (pair.first == "rec_out_diff") { continue; }
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
  std::shared_ptr<RegstDesc> rec_out_diff_regst = nullptr;
  if (parallel_ctx()->policy() == kDataParallel) {
    rec_out_diff_regst = GetProducedRegst("rec_in_diff");
  } else if (parallel_ctx()->policy() == kModelParallel) {
    rec_out_diff_regst = GetConsumedRegst("rec_out_diff");
  } else {
    UNEXPECTED_RUN();
  }
  exec_node->BindBnInOpAndRegst("rec_out_diff", rec_out_diff_regst);
}

void RecurrentBackwardCompTaskNode::VirtualBuildInDiffRegst() {
  std::shared_ptr<const Operator> op = chain_node()->SoleOp();
  ExecNode* exec_node = mut_exec_gph().SoleNode();

  exec_node->BindBnInOpAndRegst("in", GetConsumedRegst("in"));

  if (std::shared_ptr<RegstDesc> in_diff_regst = GetProducedRegst("in_diff")) {
    in_diff_regst->AddLbn(op->Lbn4BnInOp("in"));
    exec_node->BindBnInOpAndRegst("in_diff", in_diff_regst);
  }

  auto rec_in_diff_regst = GetProducedRegst("rec_in_diff");
  rec_in_diff_regst->AddLbn(op->Lbn4BnInOp("ht_1"));
  exec_node->BindBnInOpAndRegst("rec_in_diff", rec_in_diff_regst);

  if (std::shared_ptr<RegstDesc> h0_regst = GetConsumedRegst("h0")) {
    exec_node->BindBnInOpAndRegst("h0", h0_regst);
    std::shared_ptr<RegstDesc> h0_diff_regst = GetProducedRegst("h0_diff");
    h0_diff_regst->AddLbn(op->Lbn4BnInOp("h0"));
    exec_node->BindBnInOpAndRegst("h0_diff", h0_diff_regst);
  }
}

void RecurrentBackwardCompTaskNode::VirtualProduceInDiffAndBindEdge(
    TaskEdge* edge) {
  if (CanBindInDiff(edge)) {
    edge->AddRegst("in_diff", ProduceRegst("in_diff"));
  } else {
    edge->AddRegst("h0_diff", ProduceRegst("h0_diff"));
  }
}

void RecurrentBackwardCompTaskNode::VirtualProduceRegstOnRecurrentEdge(
    TaskEdge* edge) {
  edge->AddRegst("rec_in_diff", ProduceRegst("rec_in_diff", 1, 1));
}

void RecurrentBackwardCompTaskNode::VirtualConsumeDiffRegst(TaskEdge* edge) {
  std::shared_ptr<const Operator> op = chain_node()->SoleOp();
  std::shared_ptr<RegstDesc> regst = edge->GetSoleRegst();
  const auto& lbns = PredChainNodeOnEdge(edge)->data_output_lbns();
  if (lbns.find(op->Lbn4BnInOp("out_diff")) != lbns.end()) {
    ConsumeRegst("out_diff", regst);
  } else if (lbns.find(op->Lbn4BnInOp("rec_out_diff")) != lbns.end()) {
    CHECK_EQ(parallel_ctx()->policy(), kModelParallel);
    ConsumeRegst("rec_out_diff", regst);
  }
}

void RecurrentBackwardCompTaskNode::VirtualConsumeInRegst() {
  CompTaskNode* fw_node = static_cast<CompTaskNode*>(GetRelatedFwTaskNode());
  std::shared_ptr<const Operator> op = fw_node->chain_node()->SoleOp();
  for (TaskEdge* edge : fw_node->in_edges()) {
    if (edge->src_node()->GetTaskType() == TaskType::kMdUpdt) { continue; }
    std::shared_ptr<RegstDesc> regst = edge->GetSoleRegst();
    const auto& lbns = PredChainNodeOnEdge(edge)->data_output_lbns();
    if (lbns.find(op->Lbn4BnInOp("in")) != lbns.end()) {
      ConsumeRegst("in", regst);
    } else if (lbns.find(op->Lbn4BnInOp("h0")) != lbns.end()) {
      ConsumeRegst("h0", regst);
    }
  }
}

void RecurrentBackwardCompTaskNode::VirtualInferBlobDescInHiddenDiff() {
  auto rec_in_diff_regst = GetProducedRegst("rec_in_diff");
  auto rec_in_regst = GetRelatedFwTaskNode()->GetConsumedRegstWrapper("rec_in");
  rec_in_diff_regst->CopyBlobDescWithoutAddLbn(rec_in_regst.get());
  if (std::shared_ptr<RegstDesc> h0_diff_regst = GetConsumedRegst("h0")) {
    h0_diff_regst->CopyBlobDescWithoutAddLbn(GetConsumedRegst("h0").get());
  }
}

bool RecurrentBackwardCompTaskNode::CanBindInDiff(TaskEdge* edge) {
  const BackwardChainNode* succ_bw_chain_node =
      static_cast<const BackwardChainNode*>(SuccChainNodeOnEdge(edge));
  const auto& lbns = succ_bw_chain_node->fw_node()->data_output_lbns();
  std::string in_lbn = chain_node()->SoleOp()->Lbn4BnInOp("in");
  return lbns.find(in_lbn) != lbns.end();
}

}  // namespace oneflow
