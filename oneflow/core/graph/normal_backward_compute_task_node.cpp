#include "oneflow/core/graph/normal_backward_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void NormalBackwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  for (TaskEdge* edge : out_edges()) {
    TaskNode* dst_node = edge->dst_node();
    if (dst_node->GetTaskType() != TaskType::kMdDiffAcc) {
      edge->AddRegst("in_diff", ProduceRegst("in_diff"));
    } else {
      edge->AddRegst("model_diff", ProduceRegst("model_diff"));
    }
  }
  ProduceRegst("activation_diff", 1, 1);
}

void NormalBackwardCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    TaskType src_task_type = src_node->GetTaskType();
    if (IsForwardTaskType(src_task_type)) {
      ConsumeRegst("activation", edge->GetRegst("activation"));
      ConsumeRegst("data_tmp", edge->GetRegst("data_tmp"));
      ConsumeRegst("out", edge->GetRegst("out"));
    } else if (src_task_type == TaskType::kNormalMdUpdt) {
      ConsumeRegst("model", edge->GetRegst("model"));
      ConsumeRegst("model_tmp", edge->GetRegst("model_tmp"));
    } else {
      ConsumeRegst("out_diff", edge->GetSoleRegst());
    }
  }
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

void NormalBackwardCompTaskNode::BuildExecGphAndRegst() {
  BuildExecGphAndBindOutDiffRegst();
  LinkFwExecNode();
  BuildActivationDiffRegst();
  BuildInDiffRegst();
  BindModelDiffRegst();
  InferBlobDescsInProducedRegsts();
}

void NormalBackwardCompTaskNode::BuildExecGphAndBindOutDiffRegst() {
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

void NormalBackwardCompTaskNode::LinkFwExecNode() {
  CompTaskNode* fw_task = GetRelatedFwTaskNode();
  HashMap<std::string, ExecNode*> op_name2fw_exec;
  fw_task->exec_gph().ForEachNode([&](ExecNode* fw_exec) {
    CHECK(op_name2fw_exec.emplace(fw_exec->op()->op_name(), fw_exec).second);
  });
  mut_exec_gph().ForEachNode([&](ExecNode* bw_exec) {
    auto fw_exec_it = op_name2fw_exec.find(bw_exec->op()->op_name());
    if (fw_exec_it == op_name2fw_exec.end()) {
      // CHECK(bw_exec->op()->IsCloneOp());
    } else {
      bw_exec->set_fw_node(fw_exec_it->second);
    }
  });
}

void NormalBackwardCompTaskNode::BuildActivationDiffRegst() {
  std::shared_ptr<RegstDesc> activation_regst = GetSoleConsumedRegst("activation");
  auto activation_diff_regst = GetProducedRegst("activation_diff");
  mut_exec_gph().ForEachEdge([&](ExecEdge* edge) {
    if (edge->src_node()->op()->NeedExtraInDiffMemWhenBackward()
        || edge->dst_node()->op()->NeedOutWhenBackward()) {
      edge->src_node()->BindBnInOpAndRegst(edge->src_bn(), activation_diff_regst);
      edge->dst_node()->BindBnInOpAndRegst(edge->dst_bn(), activation_diff_regst);
      activation_diff_regst->AddLbi(edge->lbi());
    } else {
      edge->src_node()->BindBnInOpAndRegst(edge->src_bn(), activation_regst);
      edge->dst_node()->BindBnInOpAndRegst(edge->dst_bn(), activation_regst);
    }
    edge->src_node()->BindBnInOpAndRegst(GenUnDiffBn(edge->src_bn()), activation_regst);
    edge->dst_node()->BindBnInOpAndRegst(GenUnDiffBn(edge->dst_bn()), activation_regst);
  });
}

void NormalBackwardCompTaskNode::BuildInDiffRegst() {
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

void NormalBackwardCompTaskNode::BindModelDiffRegst() {
  std::shared_ptr<RegstDesc> data_tmp_regst = GetSoleConsumedRegst("data_tmp");
  std::shared_ptr<RegstDesc> model_tmp_regst = GetSoleConsumedRegst("model_tmp");
  std::shared_ptr<RegstDesc> model_regst = GetSoleConsumedRegst("model");
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

void NormalBackwardCompTaskNode::InferBlobDescsInProducedRegsts() {
  if (std::shared_ptr<RegstDesc> in_diff_regst = GetProducedRegst("in_diff")) {
    std::shared_ptr<RegstDesc> in_regst = GetSoleConsumedRegst("in");
    in_diff_regst->CopyBlobDescWithoutAddLbi(in_regst.get());
  }

  std::shared_ptr<RegstDesc> md_diff_regst = GetProducedRegst("model_diff");
  if (md_diff_regst) { md_diff_regst->CopyBlobDescFrom(GetSoleConsumedRegst("model").get()); }

  auto activation_diff_regst = GetProducedRegst("activation_diff");
  activation_diff_regst->CopyBlobDescWithoutAddLbi(GetSoleConsumedRegst("activation").get(),
                                                   GetSoleConsumedRegst("out").get());
}

CompTaskNode* NormalBackwardCompTaskNode::GetRelatedFwTaskNode() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* fw_node = edge->src_node();
    if (IsForwardTaskType(fw_node->GetTaskType())) { return static_cast<CompTaskNode*>(fw_node); }
  }
  return nullptr;
}

}  // namespace oneflow
