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
      if (dst_node->GetTaskType() != TodoTaskType::kMdDiffAcc) {
        edge->AddRegst("in_diff", in_diff_regst);
      }
    }
  }
  auto model_diff_regst = ProduceRegst("model_diff", 1, kMaxRegisterNum);
  for (TaskEdge* edge : out_edges()) {
    TaskNode* dst_node = edge->dst_node();
    if (dst_node->GetTaskType() == TodoTaskType::kMdDiffAcc) {
      edge->AddRegst("model_diff", model_diff_regst);
    }
  }
  ProduceRegst("activation_diff", 1, 1);
}

void BackwardCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    if (src_node->GetTaskType() == TodoTaskType::kForward) {
      ConsumeRegst("activation", edge->GetRegst("activation"));
      ConsumeRegst("data_tmp", edge->GetRegst("data_tmp"));
      ConsumeRegst("out", edge->GetRegst("out"));
    } else if (src_node->GetTaskType() == TodoTaskType::kMdUpdt) {
      ConsumeRegst("model", edge->GetRegst("model"));
    } else if (src_node->GetTaskType() == TodoTaskType::kBoxing) {
      ConsumeRegst("out_diff", edge->GetRegst("out"));
    }
    // boxing node may be deleted
    else {
      ConsumeRegst("out_diff", edge->GetRegst("in_diff"));
    }
  }
}

void BackwardCompTaskNode::BuildExecGphAndRegst() {
  Lbn2NodeBnMap lbn2producer;
  Lbn2NodeBnMap lbn2consumer;
  Lbn2NodeBnMap extern_in_lbn2consumer;
  BuildExecGphFromUserOps(&lbn2producer, &lbn2consumer,
                          &extern_in_lbn2consumer);
  SetExecNodeFromOutdiffRegst(extern_in_lbn2consumer);
  AddLbn2ActivationDiffRegst();
  AddLbn2InDiffRegst(lbn2consumer);
  AddLbn2ModelDiffRegst();
  InferBlobDescsInProducedRegsts();
}

void BackwardCompTaskNode::InferBlobDescsInProducedRegsts() {
  if (auto in_diff_regst = GetProducedRegst("in_diff")) {
    for (TaskEdge* edge : in_edges()) {
      auto src_node = edge->src_node();
      if (src_node->GetTaskType() == TodoTaskType::kForward) {
        for (TaskEdge* edge : src_node->in_edges()) {
          auto pre_src_node = edge->src_node();
          if (pre_src_node->GetTaskType() != TodoTaskType::kMdUpdt) {
            auto in_regst = edge->GetRegst("out");
            in_diff_regst->CopyBlobDescFrom(in_regst.get());
            break;
          }
        }
        break;
      }
    }
  }

  auto md_diff_regst = GetProducedRegst("model_diff");
  md_diff_regst->CopyBlobDescFrom(GetConsumedRegst("model").get());

  auto activation_diff_regst = GetProducedRegst("activation_diff");
  activation_diff_regst->CopyBlobDescFrom(GetConsumedRegst("activation").get());
}

void BackwardCompTaskNode::BuildExecGphFromUserOps(
    Lbn2NodeBnMap* lbn2producer, Lbn2NodeBnMap* lbn2consumer,
    Lbn2NodeBnMap* extern_in_lbn2consumer) {
  for (std::shared_ptr<const Operator> op : chain_node()->op_vec()) {
    ExecNode* cur_node = mut_exec_gph().NewNode();
    cur_node->mut_op() = op;
    for (const std::string& ibn : op->input_bns()) {
      const std::string& lbn = op->Lbn4BnInOp(ibn);
      CHECK(lbn2producer->insert({lbn, {cur_node, ibn}}).second);
    }
    for (const std::string& obn : op->output_bns()) {
      const std::string& lbn = op->Lbn4BnInOp(obn);
      CHECK(lbn2consumer->insert({lbn, {cur_node, obn}}).second);
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

void BackwardCompTaskNode::AddLbn2ActivationDiffRegst() {
  auto activation_regst = GetConsumedRegst("activation");
  auto activation_diff_regst = GetProducedRegst("activation_diff");
  activation_diff_regst->CopyBlobDescFrom(activation_regst.get());
  mut_exec_gph().ForEachEdge([&](ExecEdge* edge) {
    edge->src_node()->BindBnInOpAndRegst(edge->src_bn(), activation_diff_regst);
    edge->dst_node()->BindBnInOpAndRegst(edge->dst_bn(), activation_diff_regst);
    edge->src_node()->BindBnInOpAndRegst(GenUnDiffBn(edge->src_bn()),
                                         activation_regst);
    edge->dst_node()->BindBnInOpAndRegst(GenUnDiffBn(edge->dst_bn()),
                                         activation_regst);
  });
}

void BackwardCompTaskNode::SetExecNodeFromOutdiffRegst(
    const Lbn2NodeBnMap& extern_in_lbn2consumer) {
  if (extern_in_lbn2consumer.empty()) { return; }
  auto out_regst = GetConsumedRegst("out");
  auto out_diff_regst = GetConsumedRegst("out_diff");
  for (const auto& pair : extern_in_lbn2consumer) {
    ExecNode* node = pair.second.first;
    const std::string& ibn = pair.second.second;
    node->BindBnInOpAndRegst(ibn, out_diff_regst);
    node->BindBnInOpAndRegst(ibn, out_regst);
  }
}

void BackwardCompTaskNode::AddLbn2InDiffRegst(
    const Lbn2NodeBnMap& lbn2consumer) {
  auto in_diff_regst = GetProducedRegst("in_diff_regst");
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    for (const std::string& ibn : cur_node->op()->input_bns()) {
      const std::string& lbn = cur_node->op()->Lbn4BnInOp(ibn);
      if (lbn2consumer.find(lbn) == lbn2consumer.end()) {
        in_diff_regst->AddLbn(lbn);
        cur_node->BindBnInOpAndRegst(ibn, in_diff_regst);
      }
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

}  // namespace oneflow
