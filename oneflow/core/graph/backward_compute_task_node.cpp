#include "oneflow/core/graph/backward_compute_task_node.h"
#include "oneflow/core/graph/boxing_task_node.h"
#include "oneflow/core/graph/chain_graph.h"
#include "oneflow/core/graph/chain_node.h"
#include "oneflow/core/graph/forward_compute_task_node.h"
#include "oneflow/core/graph/model_diff_accumulate_compute_task_node.h"
#include "oneflow/core/graph/model_update_compute_task_node.h"

namespace oneflow {

void BackwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  bool need_in_diff = false;
  chain_node()->ForEachNodeOnOutEdge([&](const ChainNode* out_node) {
    if (dynamic_cast<const BackwardChainNode*>(out_node)) {
      need_in_diff = true;
    }
  });
  if (need_in_diff) {
    auto in_diff_regst = ProduceRegst("in_diff", 1, kMaxRegisterNum);
    for (TaskEdge* edge : out_edges()) {
      TaskNode* dst_node = edge->dst_node();
      if (!dynamic_cast<MdDiffAccCompTaskNode*>(dst_node)) {
        edge->AddRegst("in_diff", in_diff_regst);
      }
    }
  }
  auto model_diff_regst = ProduceRegst("model_diff", 1, kMaxRegisterNum);
  for (TaskEdge* edge : out_edges()) {
    TaskNode* dst_node = edge->dst_node();
    if (dynamic_cast<MdDiffAccCompTaskNode*>(dst_node)) {
      edge->AddRegst("model_diff", model_diff_regst);
    }
  }
  ProduceRegst("activation_diff", 1, 1);
}

void BackwardCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    if (dynamic_cast<ForwardCompTaskNode*>(src_node)) {
      ConsumeRegst("activation", edge->GetRegst("activation"));
      ConsumeRegst("data_tmp", edge->GetRegst("data_tmp"));
      ConsumeRegst("out", edge->GetRegst("out"));
    } else if (dynamic_cast<MdUpdtCompTaskNode*>(src_node)) {
      ConsumeRegst("model", edge->GetRegst("model"));
    } else if (dynamic_cast<BoxingTaskNode*>(src_node)) {
      ConsumeRegst("out_diff", edge->GetRegst("out"));
    }
    // boxing node may be deleted
    else {
      ConsumeRegst("out_diff", edge->GetRegst("in_diff"));
    }
  }
}

void BackwardCompTaskNode::BuildExecGphAndRegst() {
  // TODO
}

void BackwardCompTaskNode::BuildExecGphFromUserOps(
    Lbn2NodeBnMap* lbn2producer, Lbn2NodeBnMap* extern_in_lbn2consumer) {
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
      auto producer_it = lbn2producer->find(lbn);
      if (producer_it != lbn2producer->end()) {
        ExecEdge* edge = mut_exec_gph().NewEdge();
        edge->set_lbn(lbn);
        edge->mut_src_bn() = GenDiffBn(producer_it->second.second);
        edge->mut_dst_bn() = GenDiffBn(obn);
        Connect(producer_it->second.first, edge, cur_node);
      } else {
        CHECK(extern_in_lbn2consumer->insert({lbn, {cur_node, obn}}).second);
      }
    }
  });
}

void BackwardCompTaskNode::AddLbn2ActivationDiffRegst() {}

void BackwardCompTaskNode::SetExecNodeFromOutdiffRegst() {}

void BackwardCompTaskNode::AddLbn2InDiffRegst() {}

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

}  // namespace oneflow
