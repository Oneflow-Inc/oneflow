#include "oneflow/core/graph/backward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void BackwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  for (TaskEdge* edge : out_edges()) {
    TaskNode* dst_node = edge->dst_node();
    if (dst_node->GetTaskType() != TaskType::kMdDiffAcc) {
      edge->AddRegst("in_diff", ProduceRegst("in_diff", 1, kMaxRegisterNum));
    } else {
      auto model_diff_regst = ProduceRegst("model_diff", 1, kMaxRegisterNum);
      edge->AddRegst("model_diff", model_diff_regst);
    }
  }
  ProduceRegst("activation_diff", 1, 1);
}

void BackwardCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    TaskType src_task_type = src_node->GetTaskType();
    if (src_task_type == TaskType::kForward) {
      ConsumeRegst("activation", edge->GetRegst("activation"));
      ConsumeRegst("data_tmp", edge->GetRegst("data_tmp"));
      ConsumeRegst("out", edge->GetRegst("out"));
    } else if (src_task_type == TaskType::kMdUpdt) {
      ConsumeRegst("model", edge->GetRegst("model"));
      ConsumeRegst("model_tmp", edge->GetRegst("model_tmp"));
    } else {
      ConsumeRegst("out_diff", edge->GetSoleRegst());
    }
  }

  if (GetProducedRegst("in_diff")) { ConsumeRegst("in", GetRelatedInRegst()); }
}

void BackwardCompTaskNode::BuildExecGphAndRegst() {
  BuildExecGphAndBindOutDiffRegst();
  BuildActivationDiffRegst();
  BuildInDiffRegst();
  BuildModelDiffRegst();
  InferBlobDescsInProducedRegsts();
}

void BackwardCompTaskNode::BuildExecGphAndBindOutDiffRegst() {
  HashMap<std::string, std::pair<ExecNode*, std::string>> lbn2producer;
  for (std::shared_ptr<const Operator> op : chain_node()->op_vec()) {
    ExecNode* cur_node = mut_exec_gph().NewNode();
    cur_node->mut_op() = op;
    for (const std::string& idbn : op->input_diff_bns()) {
      const std::string& lbn = op->Lbn4BnInOp(idbn);
      CHECK(lbn2producer.insert({lbn, {cur_node, idbn}}).second);
    }
  }
  std::shared_ptr<RegstDesc> out_regst = GetConsumedRegst("out");
  std::shared_ptr<RegstDesc> out_diff_regst = GetConsumedRegst("out_diff");
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    for (const std::string& odbn : cur_node->op()->output_diff_bns()) {
      const std::string& lbn = cur_node->op()->Lbn4BnInOp(odbn);
      auto producer_it = lbn2producer.find(lbn);
      if (producer_it != lbn2producer.end()) {
        ExecEdge* edge = mut_exec_gph().NewEdge();
        edge->set_lbn(lbn);
        edge->mut_src_bn() = producer_it->second.second;
        edge->mut_dst_bn() = odbn;
        Connect(producer_it->second.first, edge, cur_node);
      } else {
        cur_node->BindBnInOpAndRegst(odbn, out_diff_regst);
        cur_node->BindBnInOpAndRegst(GenUnDiffBn(odbn), out_regst);
      }
    }
  });
}

void BackwardCompTaskNode::BuildActivationDiffRegst() {
  std::shared_ptr<RegstDesc> activation_regst = GetConsumedRegst("activation");
  auto activation_diff_regst = GetProducedRegst("activation_diff");
  mut_exec_gph().ForEachEdge([&](ExecEdge* edge) {
    auto type_case = edge->src_node()->op()->op_conf().op_type_case();
    if (IsDiffImplementedInPlace(type_case)) {
      edge->src_node()->BindBnInOpAndRegst(edge->src_bn(), activation_regst);
      edge->dst_node()->BindBnInOpAndRegst(edge->dst_bn(), activation_regst);
      activation_diff_regst->EraseBlobDesc(edge->lbn());
    } else {
      edge->src_node()->BindBnInOpAndRegst(edge->src_bn(),
                                           activation_diff_regst);
      edge->dst_node()->BindBnInOpAndRegst(edge->dst_bn(),
                                           activation_diff_regst);
    }

    edge->src_node()->BindBnInOpAndRegst(GenUnDiffBn(edge->src_bn()),
                                         activation_regst);
    edge->dst_node()->BindBnInOpAndRegst(GenUnDiffBn(edge->dst_bn()),
                                         activation_regst);
  });
}

void BackwardCompTaskNode::BuildInDiffRegst() {
  std::shared_ptr<RegstDesc> in_diff_regst = GetProducedRegst("in_diff");
  if (!in_diff_regst) { return; }
  std::shared_ptr<RegstDesc> in_regst = GetConsumedRegst("in");
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    HashSet<std::string> found_lbns;
    for (ExecEdge* out_edge : cur_node->out_edges()) {
      CHECK(found_lbns.insert(out_edge->lbn()).second);
    }
    for (const std::string& idbn : cur_node->op()->input_diff_bns()) {
      const std::string& lbn = cur_node->op()->Lbn4BnInOp(idbn);
      if (found_lbns.find(lbn) != found_lbns.end()) { continue; }
      in_diff_regst->AddLbn(lbn);
      cur_node->BindBnInOpAndRegst(idbn, in_diff_regst);
      cur_node->BindBnInOpAndRegst(GenUnDiffBn(idbn), in_regst);
    }
  });
}

void BackwardCompTaskNode::BuildModelDiffRegst() {
  std::shared_ptr<RegstDesc> data_tmp_regst = GetConsumedRegst("data_tmp");
  std::shared_ptr<RegstDesc> model_tmp_regst = GetConsumedRegst("model_tmp");
  std::shared_ptr<RegstDesc> model_diff_regst = GetProducedRegst("model_diff");
  std::shared_ptr<RegstDesc> model_regst = GetConsumedRegst("model");
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

  auto activation_diff_regst = GetProducedRegst("activation_diff");
  activation_diff_regst->CopyBlobDescFrom(GetConsumedRegst("activation").get());
}

std::shared_ptr<RegstDesc> BackwardCompTaskNode::GetRelatedInRegst() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* fw_node = edge->src_node();
    if (fw_node->GetTaskType() != TaskType::kForward) { continue; }
    for (TaskEdge* edge : fw_node->in_edges()) {
      TaskNode* pred_fw_node = edge->src_node();
      if (pred_fw_node->GetTaskType() != TaskType::kMdUpdt) {
        return edge->GetSoleRegst();
      }
    }
  }
  return nullptr;
}

}  // namespace oneflow
