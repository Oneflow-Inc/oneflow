#include "oneflow/core/graph/backward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void BackwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  for (TaskEdge* edge : out_edges()) {
    if (IsRecurrentOutEdge(edge)) {
      VirtualProduceRegstOnRecurrentEdge(edge);
      continue;
    }
    TaskNode* dst_node = edge->dst_node();
    if (dst_node->GetTaskType() != TaskType::kMdDiffAcc) {
      VirtualProduceInDiffAndBindEdge(edge);
    } else {
      edge->AddRegst("model_diff", ProduceRegst("model_diff"));
    }
  }
  VirtualProduceActivationDiff();
}

void BackwardCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    TaskType src_task_type = src_node->GetTaskType();
    if (IsForwardTaskType(src_task_type)) {
      VirtualConsumeActivation(edge);
      ConsumeRegst("data_tmp", edge->GetRegst("data_tmp"));
      ConsumeRegst("out", edge->GetRegst("out"));
    } else if (src_task_type == TaskType::kMdUpdt) {
      ConsumeRegst("model", edge->GetRegst("model"));
      ConsumeRegst("model_tmp", edge->GetRegst("model_tmp"));
    } else {
      VirtualConsumeDiffRegst(edge);
    }
  }
  VirtualConsumeInRegst();
}

void BackwardCompTaskNode::BuildExecGphAndRegst() {
  VirtualBuildExecGphAndBindOutDiffRegst();
  VirtualBuildActivationDiffRegst();
  VirtualBuildInDiffRegst();
  BindModelDiffRegst();
  InferBlobDescsInProducedRegsts();
}

void BackwardCompTaskNode::BindModelDiffRegst() {
  std::shared_ptr<RegstDesc> data_tmp_regst = GetConsumedRegst("data_tmp");
  std::shared_ptr<RegstDesc> model_tmp_regst = GetConsumedRegst("model_tmp");
  std::shared_ptr<RegstDesc> model_regst = GetConsumedRegst("model");
  std::shared_ptr<RegstDesc> model_diff_regst = GetProducedRegst("model_diff");
  mut_exec_gph().ForEachNode([&](ExecNode* node) {
    for (const std::string& dtbn : node->op()->data_tmp_bns()) {
      node->BindBnInOpAndRegst(dtbn, data_tmp_regst);
    }
    for (const std::string& mtbn : node->op()->model_tmp_bns()) {
      node->BindBnInOpAndRegst(mtbn, model_tmp_regst);
    }
    for (const std::string& mdbn : node->op()->model_diff_bns()) {
      node->BindBnInOpAndRegst(mdbn, model_diff_regst);
    }
    for (const std::string& mbn : node->op()->model_bns()) {
      node->BindBnInOpAndRegst(mbn, model_regst);
    }
  });
}

void BackwardCompTaskNode::InferBlobDescsInProducedRegsts() {
  if (std::shared_ptr<RegstDesc> in_diff_regst = GetProducedRegst("in_diff")) {
    std::shared_ptr<RegstDesc> in_regst = GetConsumedRegst("in");
    in_diff_regst->CopyBlobDescWithoutAddLbn(in_regst.get());
  }

  std::shared_ptr<RegstDesc> md_diff_regst = GetProducedRegst("model_diff");
  md_diff_regst->CopyBlobDescFrom(GetConsumedRegst("model").get());

  VirtualInferBlobDescInActivationDiff();
  VirtualInferBlobDescInHiddenDiff();
}

TaskNode* BackwardCompTaskNode::GetRelatedFwTaskNode() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* fw_node = edge->src_node();
    if (IsForwardTaskType(fw_node->GetTaskType())) { return fw_node; }
  }
  return nullptr;
}

}  // namespace oneflow
