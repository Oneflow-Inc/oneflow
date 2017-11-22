#include "oneflow/core/graph/backward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void BackwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  bool need_in_diff = false;
  chain_node()->ForEachNodeOnOutEdge([&](const ChainNode* out_node) {
    if (std::string(out_node->TypeName()) == "BackwardChainNode") {
      need_in_diff = true;
    }
  });
  if (need_in_diff) {
    auto in_diff_regst = ProduceRegst("in_diff", 1, kMaxRegisterNum);
    for (TaskEdge* edge : out_edges()) {
      TaskNode* dst_node = edge->dst_node();
      if (dst_node->GetTaskType() != TaskType::kMdDiffAcc) {
        edge->AddRegst("in_diff", in_diff_regst);
      }
    }
  }
  auto model_diff_regst = ProduceRegst("model_diff", 1, kMaxRegisterNum);
  for (TaskEdge* edge : out_edges()) {
    TaskNode* dst_node = edge->dst_node();
    if (dst_node->GetTaskType() == TaskType::kMdDiffAcc) {
      edge->AddRegst("model_diff", model_diff_regst);
    }
  }
  ProduceRegst("activation_diff", 1, 1);
}

void BackwardCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    const auto& src_task_type = src_node->GetTaskType();
    if (src_task_type == TaskType::kForward) {
      ConsumeRegst("activation", edge->GetRegst("activation"));
      ConsumeRegst("data_tmp", edge->GetRegst("data_tmp"));
      ConsumeRegst("out", edge->GetRegst("out"));
    } else if (src_task_type == TaskType::kMdUpdt) {
      ConsumeRegst("model", edge->GetRegst("model"));
      ConsumeRegst("model_tmp", edge->GetRegst("model_tmp"));
    } else if (src_task_type == TaskType::kBackward
               || src_task_type == TaskType::kLoss
               || src_task_type == TaskType::kBoxing || src_task_type == kCopyHd
               || src_task_type == TaskType::kCopyCommNet) {
      ConsumeRegst("out_diff", edge->GetSoleRegst());
    }
  }

  if (GetProducedRegst("in_diff")) { ConsumeRegst("in", GetRelatedInRegst()); }
}

void BackwardCompTaskNode::BuildExecGphAndRegst() {
  Lbn2NodeBnMap lbn2producer;
  BuildExecGphFromUserOps(&lbn2producer);
  SetExecNodeFromOutdiffRegst();
  AddLbn2ActivationDiffRegst();
  AddLbn2InDiffRegst(lbn2producer);
  AddLbn2ModelDiffRegst();
  InferBlobDescsInProducedRegsts();
}

void BackwardCompTaskNode::InferBlobDescsInProducedRegsts() {
  if (auto in_diff_regst = GetProducedRegst("in_diff")) {
    auto in_regst = GetConsumedRegst("in");
    in_diff_regst->CopyBlobDescFrom(in_regst.get());
  }

  auto md_diff_regst = GetProducedRegst("model_diff");
  md_diff_regst->CopyBlobDescFrom(GetConsumedRegst("model").get());

  auto activation_diff_regst = GetProducedRegst("activation_diff");
  activation_diff_regst->CopyBlobDescFrom(GetConsumedRegst("activation").get());
}

void BackwardCompTaskNode::BuildExecGphFromUserOps(
    Lbn2NodeBnMap* lbn2producer) {
  for (std::shared_ptr<const Operator> op : chain_node()->op_vec()) {
    ExecNode* cur_node = mut_exec_gph().NewNode();
    cur_node->mut_op() = op;
    for (const std::string& ibn : op->input_bns()) {
      const std::string& lbn = op->Lbn4BnInOp(ibn);
      CHECK(lbn2producer->insert({lbn, {cur_node, ibn}}).second);
    }
  }
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    for (const std::string& obn : cur_node->op()->output_bns()) {
      const std::string& lbn = cur_node->op()->Lbn4BnInOp(obn);
      const auto& producer_it = lbn2producer->find(lbn);
      if (producer_it != lbn2producer->end()) {
        ExecEdge* edge = mut_exec_gph().NewEdge();
        edge->set_lbn(lbn);
        edge->mut_src_bn() = GenDiffBn(producer_it->second.second);
        edge->mut_dst_bn() = GenDiffBn(obn);
        Connect(producer_it->second.first, edge, cur_node);
      }
    }
  });
}

void BackwardCompTaskNode::AddLbn2ActivationDiffRegst() {
  auto activation_regst = GetConsumedRegst("activation");
  auto activation_diff_regst = GetProducedRegst("activation_diff");
  mut_exec_gph().ForEachEdge([&](ExecEdge* edge) {
    edge->src_node()->BindBnInOpAndRegst(edge->src_bn(), activation_diff_regst);
    edge->dst_node()->BindBnInOpAndRegst(edge->dst_bn(), activation_diff_regst);
    edge->src_node()->BindBnInOpAndRegst(GenUnDiffBn(edge->src_bn()),
                                         activation_regst);
    edge->dst_node()->BindBnInOpAndRegst(GenUnDiffBn(edge->dst_bn()),
                                         activation_regst);
  });
}

void BackwardCompTaskNode::SetExecNodeFromOutdiffRegst() {
  auto out_regst = GetConsumedRegst("out");
  auto out_diff_regst = GetConsumedRegst("out_diff");
  mut_exec_gph().ForEachNode([&](ExecNode* bp_node) {
    HashSet<std::string> found_bns;
    for (ExecEdge* edge : bp_node->in_edges()) {
      found_bns.insert(edge->dst_bn());
    }
    for (const std::string& odbn : bp_node->op()->output_diff_bns()) {
      if (found_bns.find(odbn) != found_bns.end()) { continue; }
      bp_node->BindBnInOpAndRegst(odbn, out_diff_regst);
      bp_node->BindBnInOpAndRegst(GenUnDiffBn(odbn), out_regst);
    }
  });
}

void BackwardCompTaskNode::AddLbn2InDiffRegst(
    const Lbn2NodeBnMap& lbn2producer) {
  auto in_diff_regst = GetProducedRegst("in_diff");
  if (!in_diff_regst) { return; }
  auto in_regst = GetConsumedRegst("in");
  mut_exec_gph().ForEachNode([&](ExecNode* bp_node) {
    HashSet<std::string> found_bns;
    for (ExecEdge* edge : bp_node->out_edges()) {
      found_bns.insert(edge->src_bn());
    }
    for (const std::string& idbn : bp_node->op()->input_diff_bns()) {
      if (found_bns.find(idbn) != found_bns.end()) { continue; }
      const std::string& lbn = bp_node->op()->Lbn4BnInOp(idbn);
      in_diff_regst->AddLbn(lbn);
      bp_node->BindBnInOpAndRegst(idbn, in_diff_regst);
      bp_node->BindBnInOpAndRegst(GenUnDiffBn(idbn), in_regst);
    }
  });
}

void BackwardCompTaskNode::AddLbn2ModelDiffRegst() {
  auto data_tmp_regst = GetConsumedRegst("data_tmp");
  auto model_tmp_regst = GetConsumedRegst("model_tmp");
  auto model_diff_regst = GetProducedRegst("model_diff");
  auto model_regst = GetConsumedRegst("model");
  mut_exec_gph().ForEachNode([&](ExecNode* node) {
    for (const std::string& dtbn : node->op()->data_tmp_bns()) {
      node->BindBnInOpAndRegst(dtbn, data_tmp_regst);
    }
    for (const std::string& mtbn : node->op()->model_tmp_bns()) {
      node->BindBnInOpAndRegst(mtbn, model_tmp_regst);
    }
    for (const std::string& mdbn : node->op()->model_diff_bns()) {
      const std::string& lbn = node->op()->Lbn4BnInOp(mdbn);
      model_diff_regst->AddLbn(lbn);
      node->BindBnInOpAndRegst(mdbn, model_diff_regst);
    }
    for (const std::string& mbn : node->op()->model_bns()) {
      node->BindBnInOpAndRegst(mbn, model_regst);
    }
  });
}

std::shared_ptr<RegstDesc> BackwardCompTaskNode::GetRelatedInRegst() {
  std::shared_ptr<RegstDesc> in_regst = nullptr;
  for (TaskEdge* edge : in_edges()) {
    auto src_node = edge->src_node();
    if (src_node->GetTaskType() == TaskType::kForward) {
      for (TaskEdge* edge : src_node->in_edges()) {
        auto pre_src_node = edge->src_node();
        if (pre_src_node->GetTaskType() != TaskType::kMdUpdt) {
          in_regst = edge->GetSoleRegst();
          break;
        }
      }
      break;
    }
  }
  return in_regst;
}

}  // namespace oneflow
