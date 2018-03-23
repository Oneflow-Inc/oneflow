#include "oneflow/core/graph/normalization_forward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void NormalizationForwardCompTaskNode::VirtualConsumeRegstOnInEdge(
    TaskEdge* edge) {
  if (edge->src_node()->GetTaskType() == TaskType::kNormalizationMdUpdt) {
    ConsumeRegst("norm_model", edge->GetSoleRegst());
  } else {
    ConsumeRegst("in", edge->GetSoleRegst());
  }
}

void NormalizationForwardCompTaskNode::VirtualProduceRegstOnOutEdge(
    TaskEdge* edge) {
  if (edge->dst_node()->GetTaskType() == TaskType::kNormalizationMdUpdt) {
    edge->AddRegst("norm_acc", ProduceRegst("norm_acc"));
  } else {
    edge->AddRegst("out", GetProducedRegst("out"));
    if (IsBackwardTaskType(edge->dst_node()->GetTaskType())) {
      edge->AddRegst("activation", ProduceRegst("activation"));
      edge->AddRegst("data_tmp", ProduceRegst("data_tmp"));
    }
  }
}

void NormalizationForwardCompTaskNode::
    VirtualBuildExecGphStructAndBindInRegst() {
  std::shared_ptr<RegstDesc> in_regst = GetConsumedRegst("in");
  ExecNode* cur_node = mut_exec_gph().NewNode();
  cur_node->mut_op() = chain_node()->SoleOp();
  for (const std::string& ibn : cur_node->op()->input_bns()) {
    cur_node->BindBnInOpAndRegst(ibn, in_regst);
  }
}

void NormalizationForwardCompTaskNode::VirtualBuildOutRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  ExecNode* cur_node = mut_exec_gph().SoleNode();
  for (const std::string& obn : cur_node->op()->output_bns()) {
    const std::string& lbn = cur_node->op()->Lbn4BnInOp(obn);
    out_regst->AddLbn(lbn);
    cur_node->BindBnInOpAndRegst(obn, out_regst);
  }
}

bool NormalizationForwardCompTaskNode::IsReadyForBuild() {
  return GetConsumedRegst("in")->IsLocked();
}

void NormalizationForwardCompTaskNode::VirtualBuildExtraRegsts() {
  std::shared_ptr<RegstDesc> norm_model_regst = GetConsumedRegst("norm_model");
  std::shared_ptr<RegstDesc> norm_acc_regst = GetProducedRegst("norm_acc");
  ExecNode* node = mut_exec_gph().SoleNode();
  std::vector<std::string> norm_model_bns = {"moving_mean", "moving_variance"};
  for (const std::string bn : norm_model_bns) {
    norm_model_regst->AddLbn(node->op()->Lbn4BnInOp(bn));
    node->BindBnInOpAndRegst(bn, norm_model_regst);
  }
  if (norm_acc_regst) {
    std::vector<std::string> norm_acc_bns = {"new_mean", "new_variance"};
    for (const std::string bn : norm_acc_bns) {
      norm_acc_regst->AddLbn(node->op()->Lbn4BnInOp(bn));
      node->BindBnInOpAndRegst(bn, norm_acc_regst);
    }
  }
}

void NormalizationForwardCompTaskNode::VirtualLockExtraRegsts() {
  TryLockConsumedRegst("norm_model");
}

}  // namespace oneflow
