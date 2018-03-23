#include "oneflow/core/graph/forward_compute_task_node.h"
#include "oneflow/core/graph/normal_forward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void NormalForwardCompTaskNode::VirtualConsumeRegstOnInEdge(TaskEdge* edge) {
  ConsumeRegst("in", edge->GetSoleRegst());
}

void NormalForwardCompTaskNode::VirtualBuildExecGphStructAndBindInRegst() {
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

void NormalForwardCompTaskNode::VirtualBuildOutRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    HashSet<std::string> found_lbns;
    for (ExecEdge* out_edge : cur_node->out_edges()) {
      found_lbns.insert(out_edge->lbn());
    }
    for (const std::string& obn : cur_node->op()->output_bns()) {
      const std::string& lbn = cur_node->op()->Lbn4BnInOp(obn);
      if (found_lbns.find(lbn) != found_lbns.end()) { continue; }
      out_regst->AddLbn(lbn);
      cur_node->BindBnInOpAndRegst(obn, out_regst);
    }
  });
}

bool NormalForwardCompTaskNode::IsReadyForBuild() {
  return GetConsumedRegst("in")->IsLocked();
}

}  // namespace oneflow
