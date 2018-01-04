#include "oneflow/core/graph/backward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void BackwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  for (TaskEdge* edge : out_edges()) {
    TaskNode* dst_node = edge->dst_node();
    if (dst_node == this) {
      edge->AddRegst("ht_1_diff", ProduceRegst("ht_1_diff"));
    } else if (dst_node->GetTaskType() != TaskType::kMdDiffAcc) {
      // TODO h0_diff
      edge->AddRegst("in_diff", ProduceRegst("in_diff"));
    } else {
      edge->AddRegst("model_diff", ProduceRegst("model_diff"));
    }
  }
  if (GetTaskType() == TaskType::kNonRecurrentBackward) {
    ProduceRegst("activation_diff", 1, 1);
  }
}

void BackwardCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    TaskType src_task_type = src_node->GetTaskType();
    if (IsForwardTaskType(src_task_type)) {
      if (GetTaskType() == TaskType::kNonRecurrentBackward) {
        ConsumeRegst("activation", edge->GetRegst("activation"));
      }
      ConsumeRegst("data_tmp", edge->GetRegst("data_tmp"));
      ConsumeRegst("out", edge->GetRegst("out"));
    } else if (src_task_type == TaskType::kMdUpdt) {
      ConsumeRegst("model", edge->GetRegst("model"));
      ConsumeRegst("model_tmp", edge->GetRegst("model_tmp"));
    } else if (src_node == this) {
      ConsumeRegst("ht_1_diff", edge->GetSoleRegst());
    } else {
      ConsumeRegst("out_diff", edge->GetSoleRegst());
    }
  }

  VirtualConsumeInRegst();
}

void BackwardCompTaskNode::BuildExecGphAndRegst() {
  BuildExecGphAndBindOutDiffRegst();
  if (GetTaskType() == TaskType::kNonRecurrentBackward) {
    BuildActivationDiffRegst();
  }
  BuildInDiffRegst();
  BuildModelDiffRegst();
  InferBlobDescsInProducedRegsts();
}

void BackwardCompTaskNode::FixRegisterNumRange() {
  int32_t max_seq_size = GetConsumedRegst("in")->MaxSeqSize();
  std::shared_ptr<RegstDesc> ht_1_diff_regst = GetProducedRegst("ht_1_diff");
  std::shared_ptr<RegstDesc> in_diff_regst = GetProducedRegst("in_diff");
  ht_1_diff_regst->set_min_register_num(max_seq_size);
  in_diff_regst->set_min_register_num(max_seq_size);
}

void BackwardCompTaskNode::BuildActivationDiffRegst() {
  std::shared_ptr<RegstDesc> activation_regst = GetConsumedRegst("activation");
  auto activation_diff_regst = GetProducedRegst("activation_diff");
  mut_exec_gph().ForEachEdge([&](ExecEdge* edge) {
    if (edge->src_node()->op()->NeedExtraInDiffMemWhenBackward()
        || edge->dst_node()->op()->NeedOutWhenBackward()) {
      edge->src_node()->BindBnInOpAndRegst(edge->src_bn(),
                                           activation_diff_regst);
      edge->dst_node()->BindBnInOpAndRegst(edge->dst_bn(),
                                           activation_diff_regst);
      activation_diff_regst->AddLbn(edge->lbn());
    } else {
      edge->src_node()->BindBnInOpAndRegst(edge->src_bn(), activation_regst);
      edge->dst_node()->BindBnInOpAndRegst(edge->dst_bn(), activation_regst);
    }

    edge->src_node()->BindBnInOpAndRegst(GenUnDiffBn(edge->src_bn()),
                                         activation_regst);
    edge->dst_node()->BindBnInOpAndRegst(GenUnDiffBn(edge->dst_bn()),
                                         activation_regst);
  });
}

void BackwardCompTaskNode::BuildModelDiffRegst() {
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

  if (GetTaskType() == TaskType::kNonRecurrentBackward) {
    auto activation_diff_regst = GetProducedRegst("activation_diff");
    activation_diff_regst->CopyBlobDescWithoutAddLbn(
        GetConsumedRegst("activation").get());
  } else {
    auto ht_1_diff_regst = GetProducedRegst("ht_1_diff");
    ht_1_diff_regst->CopyBlobDescWithoutAddLbn(GetConsumedRegst("out").get());
  }

  if (GetConsumedRegst("h0")) {
    auto h0_diff_regst = GetProducedRegst("h0_diff_regst");
    h0_diff_regst->CopyBlobDescWithoutAddLbn(GetConsumedRegst("h0").get());
  }
}

TaskNode* BackwardCompTaskNode::GetRelatedFwTaskNode() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* fw_node = edge->src_node();
    if (IsForwardTaskType(fw_node->GetTaskType())) { return fw_node; }
  }
  return nullptr;
}

}  // namespace oneflow
