#include "oneflow/core/graph/normal_backward_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void NormalBackwardCompTaskNode::VirtualBuildExecGphAndBindOutDiffRegst() {
  HashMap<LogicalBlobId, std::pair<ExecNode*, std::string>> lbi2producer;
  for (std::shared_ptr<const Operator> op : logical_node()->op_vec()) {
    ExecNode* cur_node = mut_exec_gph().NewNode();
    cur_node->mut_op() = op;
    for (const std::string& idbn : op->input_diff_bns()) {
      const LogicalBlobId& lbi = op->BnInOp2Lbi(idbn);
      CHECK(lbi2producer.insert({lbi, {cur_node, idbn}}).second);
    }
  }
  std::shared_ptr<RegstDesc> out_regst = GetSoleConsumedRegst("out");
  std::shared_ptr<RegstDesc> out_diff_regst = GetSoleConsumedRegst("out_diff");
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    for (const std::string& odbn : cur_node->op()->output_diff_bns()) {
      const LogicalBlobId& lbi = cur_node->op()->BnInOp2Lbi(odbn);
      auto producer_it = lbi2producer.find(lbi);
      if (producer_it != lbi2producer.end()) {
        ExecEdge* edge = mut_exec_gph().NewEdge();
        edge->set_lbi(lbi);
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

void NormalBackwardCompTaskNode::VirtualBuildActivationDiffRegst() {
  std::shared_ptr<RegstDesc> activation_regst =
      GetSoleConsumedRegst("activation");
  auto activation_diff_regst = GetProducedRegst("activation_diff");
  mut_exec_gph().ForEachEdge([&](ExecEdge* edge) {
    if (edge->src_node()->op()->NeedExtraInDiffMemWhenBackward()
        || edge->dst_node()->op()->NeedOutWhenBackward()) {
      edge->src_node()->BindBnInOpAndRegst(edge->src_bn(),
                                           activation_diff_regst);
      edge->dst_node()->BindBnInOpAndRegst(edge->dst_bn(),
                                           activation_diff_regst);
      activation_diff_regst->AddLbi(edge->lbi());
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

void NormalBackwardCompTaskNode::VirtualBuildInDiffRegst() {
  std::shared_ptr<RegstDesc> in_diff_regst = GetProducedRegst("in_diff");
  std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in");
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    HashSet<LogicalBlobId> found_lbis;
    for (ExecEdge* out_edge : cur_node->out_edges()) {
      CHECK(found_lbis.insert(out_edge->lbi()).second);
    }
    for (const std::string& idbn : cur_node->op()->input_diff_bns()) {
      const LogicalBlobId& lbi = cur_node->op()->BnInOp2Lbi(idbn);
      if (found_lbis.find(lbi) != found_lbis.end()) { continue; }
      if (in_diff_regst) {
        in_diff_regst->AddLbi(lbi);
        cur_node->BindBnInOpAndRegst(idbn, in_diff_regst);
      }
      cur_node->BindBnInOpAndRegst(GenUnDiffBn(idbn), in_regst);
    }
  });
}

void NormalBackwardCompTaskNode::VirtualConsumeRegstOnInEdge(TaskEdge* edge) {
  ConsumeRegst("out_diff", edge->GetSoleRegst());
}

void NormalBackwardCompTaskNode::VirtualProduceInDiffAndBindEdge(
    TaskEdge* edge) {
  edge->AddRegst("in_diff", ProduceRegst("in_diff"));
}

void NormalBackwardCompTaskNode::VirtualProduceActivationDiff() {
  ProduceRegst("activation_diff", 1, 1);
}

void NormalBackwardCompTaskNode::VirtualConsumeActivation(TaskEdge* edge) {
  ConsumeRegst("activation", edge->GetRegst("activation"));
}

void NormalBackwardCompTaskNode::VirtualInferBlobDescInActivationDiff() {
  auto activation_diff_regst = GetProducedRegst("activation_diff");
  activation_diff_regst->CopyBlobDescWithoutAddLbi(
      GetSoleConsumedRegst("activation").get(),
      GetSoleConsumedRegst("out").get());
}

void NormalBackwardCompTaskNode::VirtualConsumeInRegst() {
  TaskNode* fw_node = GetRelatedFwTaskNode();
  for (TaskEdge* edge : fw_node->in_edges()) {
    TaskNode* pred_fw_node = edge->src_node();
    if (!IsMdUpdtTaskType(pred_fw_node->GetTaskType())) {
      ConsumeRegst("in", edge->GetSoleRegst());
      return;
    }
  }
  UNIMPLEMENTED();
}

}  // namespace oneflow
