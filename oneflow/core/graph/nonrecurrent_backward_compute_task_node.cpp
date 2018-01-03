#include "oneflow/core/graph/nonrecurrent_backward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void NonRecurrentBackwardCompTaskNode::BuildExecGphAndBindOutDiffRegst() {
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

void NonRecurrentBackwardCompTaskNode::BuildInDiffRegst() {
  std::shared_ptr<RegstDesc> in_diff_regst = GetProducedRegst("in_diff");
  std::shared_ptr<RegstDesc> in_regst = GetConsumedRegst("in");
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    HashSet<std::string> found_lbns;
    for (ExecEdge* out_edge : cur_node->out_edges()) {
      CHECK(found_lbns.insert(out_edge->lbn()).second);
    }
    for (const std::string& idbn : cur_node->op()->input_diff_bns()) {
      const std::string& lbn = cur_node->op()->Lbn4BnInOp(idbn);
      if (found_lbns.find(lbn) != found_lbns.end()) { continue; }
      if (in_diff_regst) {
        in_diff_regst->AddLbn(lbn);
        cur_node->BindBnInOpAndRegst(idbn, in_diff_regst);
      }
      cur_node->BindBnInOpAndRegst(GenUnDiffBn(idbn), in_regst);
    }
  });
}

void NonRecurrentBackwardCompTaskNode::VirtualConsumeInRegst() {
  TaskNode* fw_node = GetRelatedFwTaskNode();
  for (TaskEdge* edge : fw_node->in_edges()) {
    TaskNode* pred_fw_node = edge->src_node();
    if (pred_fw_node->GetTaskType() != TaskType::kMdUpdt) {
      ConsumeRegst("in", edge->GetSoleRegst());
      return;
    }
  }
  UNEXPECTED_RUN();
}

}  // namespace oneflow
