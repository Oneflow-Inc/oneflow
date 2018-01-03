#include "oneflow/core/graph/forward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void ForwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  // if (static_cast<const ForwardChainNode*>(chain_node())->bw_node()) {}
  std::shared_ptr<RegstDesc> activation_regst = ProduceRegst("activation");
  std::shared_ptr<RegstDesc> data_tmp_regst = ProduceRegst("data_tmp");
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out");
  for (TaskEdge* edge : out_edges()) {
    TaskNode* dst_node = edge->dst_node();
    if (IsBackwardTaskType(dst_node->GetTaskType())) {
      edge->AddRegst("activation", activation_regst);
      edge->AddRegst("data_tmp", data_tmp_regst);
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
      VirtualConsumeInRegst(edge);
    }
  }
}

void ForwardCompTaskNode::BuildExecGphAndRegst() {
  BuildExecGphStructAndBindInRegst();
  BuildOutRegst();
  BuildActivationRegst();
  BuildModelAndTmpRegsts();
  mut_exec_gph().TopoForEachNode([this](ExecNode* node) {
    node->op()->InferBlobDescs(node->GetBlobDesc4BnInOpFunc(), parallel_ctx());
  });
}

void ForwardCompTaskNode::BuildExecGphStructAndBindInRegst() {
  HashMap<std::string, std::pair<ExecNode*, std::string>> lbn2producer;
  for (std::shared_ptr<const Operator> op : chain_node()->op_vec()) {
    ExecNode* cur_node = mut_exec_gph().NewNode();
    cur_node->mut_op() = op;
    for (const std::string& obn : op->output_bns()) {
      const std::string& lbn = op->Lbn4BnInOp(obn);
      CHECK(lbn2producer.insert({lbn, {cur_node, obn}}).second);
    }
  }
  std::shared_ptr<RegstDesc> in_regst = GetConsumedRegst("in");
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    for (const std::string& ibn : cur_node->op()->input_bns()) {
      const std::string& lbn = cur_node->op()->Lbn4BnInOp(ibn);
      auto producer_it = lbn2producer.find(lbn);
      if (producer_it != lbn2producer.end()) {
        ExecEdge* edge = mut_exec_gph().NewEdge();
        edge->set_lbn(lbn);
        edge->mut_src_bn() = producer_it->second.second;
        edge->mut_dst_bn() = ibn;
        Connect(producer_it->second.first, edge, cur_node);
      } else {
        cur_node->BindBnInOpAndRegst(ibn, in_regst);
      }
    }
  });
}

void ForwardCompTaskNode::BuildOutRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    HashSet<std::string> found_lbns;
    for (ExecEdge* out_edge : cur_node->out_edges()) {
      CHECK(found_lbns.insert(out_edge->lbn()).second);
    }
    for (const std::string& obn : cur_node->op()->output_bns()) {
      const std::string& lbn = cur_node->op()->Lbn4BnInOp(obn);
      if (found_lbns.find(lbn) != found_lbns.end()) { continue; }
      out_regst->AddLbn(lbn);
      cur_node->BindBnInOpAndRegst(obn, out_regst);
    }
  });
}

void ForwardCompTaskNode::BuildActivationRegst() {
  std::shared_ptr<RegstDesc> activation_regst = GetProducedRegst("activation");
  mut_exec_gph().ForEachEdge([&](const ExecEdge* edge) {
    activation_regst->AddLbn(edge->lbn());
    edge->src_node()->BindBnInOpAndRegst(edge->src_bn(), activation_regst);
    edge->dst_node()->BindBnInOpAndRegst(edge->dst_bn(), activation_regst);
  });
}

void ForwardCompTaskNode::BuildModelAndTmpRegsts() {
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
  TryLockConsumedRegst("model");
  TryLockConsumedRegst("model_tmp");
}

}  // namespace oneflow
