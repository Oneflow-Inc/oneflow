#include "oneflow/core/graph/forward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void ForwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  if (static_cast<const ForwardChainNode*>(chain_node())->bw_node()) {
    ProduceRegst("activation", 1, kMaxRegisterNum);
    ProduceRegst("data_tmp", 1, kMaxRegisterNum);
  } else {
    ProduceRegst("activation", 1, 1);
    ProduceRegst("data_tmp", 1, 1);
  }

  auto out_regst = ProduceRegst("out", 1, kMaxRegisterNum);
  for (TaskEdge* edge : out_edges()) {
    TaskNode* dst_node = edge->dst_node();
    if (dst_node->GetTaskType() == TaskType::kBackward) {
      edge->AddRegst("activation", GetProducedRegst("activation"));
      edge->AddRegst("data_tmp", GetProducedRegst("data_tmp"));
    }
    edge->AddRegst("out", out_regst);
  }
}

void ForwardCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    if (src_node->GetTaskType() == TaskType::kMdUpdt) {
      ConsumeRegst("model", edge->GetRegst("model"));
      ConsumeRegst("model_tmp", edge->GetRegst("model_tmp"));
    } else {
      ConsumeRegst("in", edge->GetSoleRegst());
    }
  }
}

void ForwardCompTaskNode::BuildExecGphAndRegst() {
  Lbn2NodeBnMap lbn2producer;
  Lbn2NodeBnMap extern_in_lbn2consumer;
  BuildFromUserOps(&lbn2producer, &extern_in_lbn2consumer);
  SetExecNodeFromInRegst(extern_in_lbn2consumer);
  AddLbn2OutRegst(lbn2producer);
  AddLbn2ActivationRegst();
  AddLbn2ModelAndTmpRegsts();

  mut_exec_gph().TopoForEachNode([this](ExecNode* node) {
    node->op()->InferBlobDescs(node->GetBlobDesc4BnInOpFunc(), parallel_ctx());
  });
}

void ForwardCompTaskNode::BuildFromUserOps(
    Lbn2NodeBnMap* lbn2producer, Lbn2NodeBnMap* extern_in_lbn2consumer) {
  for (std::shared_ptr<const Operator> op : chain_node()->op_vec()) {
    ExecNode* cur_node = mut_exec_gph().NewNode();
    cur_node->mut_op() = op;
    for (const std::string& obn : op->output_bns()) {
      const std::string& lbn = op->Lbn4BnInOp(obn);
      CHECK(lbn2producer->insert({lbn, {cur_node, obn}}).second);
    }
  }
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    for (const std::string& ibn : cur_node->op()->input_bns()) {
      const std::string& lbn = cur_node->op()->Lbn4BnInOp(ibn);
      const auto& producer_it = lbn2producer->find(lbn);
      if (producer_it != lbn2producer->end()) {
        ExecEdge* edge = mut_exec_gph().NewEdge();
        edge->set_lbn(lbn);
        edge->mut_src_bn() = producer_it->second.second;
        edge->mut_dst_bn() = ibn;
        Connect(producer_it->second.first, edge, cur_node);
      } else {
        CHECK(extern_in_lbn2consumer->insert({lbn, {cur_node, ibn}}).second);
      }
    }
  });
}

void ForwardCompTaskNode::SetExecNodeFromInRegst(
    const Lbn2NodeBnMap& extern_in_lbn2consumer) {
  if (extern_in_lbn2consumer.empty()) { return; }
  std::shared_ptr<RegstDesc> in_regst = GetConsumedRegst("in");
  for (const auto& pair : extern_in_lbn2consumer) {
    ExecNode* node = pair.second.first;
    const std::string& ibn = pair.second.second;
    node->BindBnInOpAndRegst(ibn, in_regst);
  }
}

void ForwardCompTaskNode::AddLbn2OutRegst(const Lbn2NodeBnMap& lbn2producer) {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");

  for (const std::string& lbn : chain_node()->data_output_lbns()) {
    const std::pair<ExecNode*, std::string>& producer = lbn2producer.at(lbn);
    ExecNode* node = producer.first;
    const std::string& obn = producer.second;
    out_regst->AddLbn(lbn);
    node->BindBnInOpAndRegst(obn, out_regst);
  }
}

void ForwardCompTaskNode::AddLbn2ActivationRegst() {
  std::shared_ptr<RegstDesc> activation_regst = GetProducedRegst("activation");
  mut_exec_gph().ForEachEdge([&](const ExecEdge* edge) {
    activation_regst->AddLbn(edge->lbn());
    edge->src_node()->BindBnInOpAndRegst(edge->src_bn(), activation_regst);
    edge->dst_node()->BindBnInOpAndRegst(edge->dst_bn(), activation_regst);
  });

  const auto& data_output_lbns = chain_node()->data_output_lbns();
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    for (const std::string& obn : cur_node->op()->output_bns()) {
      const std::string& lbn = cur_node->op()->Lbn4BnInOp(obn);
      if (data_output_lbns.find(lbn) == data_output_lbns.end()) {
        activation_regst->AddLbn(lbn);
        cur_node->BindBnInOpAndRegst(obn, activation_regst);
      }
    }
  });
}

void ForwardCompTaskNode::AddLbn2ModelAndTmpRegsts() {
  std::shared_ptr<RegstDesc> data_tmp_regst = GetProducedRegst("data_tmp");
  std::shared_ptr<RegstDesc> model_regst = GetConsumedRegst("model");
  std::shared_ptr<RegstDesc> model_tmp_regst = GetConsumedRegst("model_tmp");
  mut_exec_gph().ForEachNode([&](ExecNode* node) {
    for (const std::string& dtbn : node->op()->data_tmp_bns()) {
      const std::string& lbn = node->op()->Lbn4BnInOp(dtbn);
      data_tmp_regst->AddLbn(lbn);
      node->BindBnInOpAndRegst(dtbn, data_tmp_regst);
    }
    for (const std::string& mtbn : node->op()->model_tmp_bns()) {
      const std::string& lbn = node->op()->Lbn4BnInOp(mtbn);
      model_tmp_regst->AddLbn(lbn);
      node->BindBnInOpAndRegst(mtbn, model_tmp_regst);
    }
    for (const std::string& mbn : node->op()->model_bns()) {
      const std::string& lbn = node->op()->Lbn4BnInOp(mbn);
      model_regst->AddLbn(lbn);
      node->BindBnInOpAndRegst(mbn, model_regst);
    }
  });
}

void ForwardCompTaskNode::LockRegsts() {
  TaskNode::LockRegsts();
  GetConsumedRegst("model")->Lock();
  GetConsumedRegst("model_tmp")->Lock();
}

bool ForwardCompTaskNode::IsReadyForBuild() {
  return GetConsumedRegst("in")->IsLocked();
}

}  // namespace oneflow
