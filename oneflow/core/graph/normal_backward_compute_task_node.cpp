#include "oneflow/core/graph/normal_backward_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void NormalBackwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceB121Regst("in_diff");
  ProduceRegst("activation_diff", 1, 1);
  for (TaskEdge* edge : out_edges()) {
    const LogicalNode* succ_logical = GetOneSuccLogicalNodeOnEdge(edge);
    if (succ_logical->TypeName() == "MdDiffAcc" || succ_logical->TypeName() == "NormalMdUpdt"
        || succ_logical->TypeName() == "ReduceScatter") {
      edge->AddRegst("model_diff", ProduceRegst("model_diff"));
    } else {
      BindEdgeWithProducedB121Regst(edge, "in_diff");
    }
  }
}

void NormalBackwardCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    TaskType src_task_type = src_node->GetTaskType();
    if (IsForwardTaskType(src_task_type)) {
      ConsumeRegst("activation", edge->GetRegst("activation"));
      ConsumeRegst("data_tmp", edge->GetRegst("data_tmp"));
      ConsumeRegst("boxing_out", edge->GetRegst("boxing_out"));
      ConsumeRegst("121_out", edge->GetRegst("121_out"));
      ConsumeRegst("const_buf", edge->GetRegst("const_buf"));
    } else if (src_task_type == TaskType::kNormalMdUpdt) {
      ConsumeRegst("model", edge->GetRegst("model"));
      ConsumeRegst("const_model", edge->GetRegst("const_model"));
    } else {
      ConsumeRegst("out_diff", edge->GetSoleRegst());
    }
  }
  CompTaskNode* fw_task = GetRelatedFwTaskNode();
  if (fw_task) {
    const std::list<std::weak_ptr<RegstDesc>>& in_regst = fw_task->GetConsumedRegst("in");
    for (std::weak_ptr<RegstDesc> regst : in_regst) { ConsumeRegst("in", regst.lock()); }
  }
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
        cur_node->BindBnWithOneOfTheRegsts(odbn, GetConsumedRegst("out_diff"));
      }
    }
  });
  CompTaskNode* fw_task = GetRelatedFwTaskNode();
  if (fw_task) {
    const HashSet<LogicalBlobId>& lbi_boxing = fw_task->logical_node()->lbi_boxing();
    const HashSet<LogicalBlobId>& lbi_121 = fw_task->logical_node()->lbi_121();
    std::shared_ptr<RegstDesc> out_regst_boxing = GetSoleConsumedRegst("boxing_out");
    std::shared_ptr<RegstDesc> out_regst_121 = GetSoleConsumedRegst("121_out");
    mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
      for (const std::string& odbn : cur_node->op()->output_diff_bns()) {
        const LogicalBlobId& lbi = cur_node->op()->BnInOp2Lbi(odbn);
        if (lbi2producer.find(lbi) == lbi2producer.end()) {
          if (lbi_boxing.find(lbi) != lbi_boxing.end()) {
            cur_node->BindBnWithRegst(GenUnDiffBn(odbn), out_regst_boxing);
          } else if (lbi_121.find(lbi) != lbi_121.end()) {
            cur_node->BindBnWithRegst(GenUnDiffBn(odbn), out_regst_121);
          } else {
            UNIMPLEMENTED();
          }
        }
      }
    });
  }
}

void NormalBackwardCompTaskNode::LinkFwExecNode() {
  CompTaskNode* fw_task = GetRelatedFwTaskNode();
  if (fw_task == nullptr) { return; }
  HashMap<std::string, ExecNode*> op_name2fw_exec;
  fw_task->exec_gph().ForEachNode([&](ExecNode* fw_exec) {
    CHECK(op_name2fw_exec.emplace(fw_exec->op()->op_name(), fw_exec).second);
  });
  mut_exec_gph().ForEachNode([&](ExecNode* bw_exec) {
    auto fw_exec_it = op_name2fw_exec.find(bw_exec->op()->op_name());
    if (fw_exec_it != op_name2fw_exec.end()) { bw_exec->set_fw_node(fw_exec_it->second); }
  });
}

void NormalBackwardCompTaskNode::BuildActivationDiffRegst() {
  std::shared_ptr<RegstDesc> activation_regst = GetSoleConsumedRegst("activation");
  std::shared_ptr<RegstDesc> activation_diff_regst = GetProducedRegst("activation_diff");
  mut_exec_gph().ForEachEdge([&](ExecEdge* edge) {
    if (edge->src_node()->op()->NeedExtraInDiffMemWhenBackward()
        || edge->dst_node()->op()->NeedOutWhenBackward() || edge->src_node()->fw_node() == nullptr
        || edge->dst_node()->fw_node() == nullptr) {
      edge->src_node()->BindBnWithRegst(edge->src_bn(), activation_diff_regst);
      edge->dst_node()->BindBnWithRegst(edge->dst_bn(), activation_diff_regst);
      activation_diff_regst->AddLbi(edge->lbi());
    } else {
      edge->src_node()->BindBnWithRegst(edge->src_bn(), activation_regst);
      edge->dst_node()->BindBnWithRegst(edge->dst_bn(), activation_regst);
    }
    edge->src_node()->BindBnWithRegst(GenUnDiffBn(edge->src_bn()), activation_regst);
    edge->dst_node()->BindBnWithRegst(GenUnDiffBn(edge->dst_bn()), activation_regst);
  });
}

void NormalBackwardCompTaskNode::BuildInDiffRegst() {
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    HashSet<LogicalBlobId> found_lbis;
    for (ExecEdge* out_edge : cur_node->out_edges()) {
      CHECK(found_lbis.insert(out_edge->lbi()).second);
    }
    for (const std::string& idbn : cur_node->op()->input_diff_bns()) {
      const LogicalBlobId& lbi = cur_node->op()->BnInOp2Lbi(idbn);
      if (GetRelatedFwTaskNode()) {
        cur_node->BindBnWithOneOfTheRegsts(GenUnDiffBn(idbn), GetConsumedRegst("in"));
      }
      if (TryAddLbiToB121RegstAndBindIt(cur_node, idbn, "in_diff") == false) {
        CHECK(found_lbis.empty() || found_lbis.find(lbi) != found_lbis.end());
      }
    }
  });
}

void NormalBackwardCompTaskNode::BindModelDiffRegst() {
  std::shared_ptr<RegstDesc> data_tmp_regst = GetSoleConsumedRegst("data_tmp");
  std::shared_ptr<RegstDesc> const_model_regst = GetSoleConsumedRegst("const_model");
  std::shared_ptr<RegstDesc> model_regst = GetSoleConsumedRegst("model");
  std::shared_ptr<RegstDesc> const_buf_regst = GetSoleConsumedRegst("const_buf");
  std::shared_ptr<RegstDesc> model_diff_regst = GetProducedRegst("model_diff");
  mut_exec_gph().ForEachNode([&](ExecNode* node) {
    node->BindBnsWithRegst(&Operator::data_tmp_bns, data_tmp_regst);
    node->BindBnsWithRegst(&Operator::const_model_bns, const_model_regst);
    node->BindBnsWithRegst(&Operator::const_buf_bns, const_buf_regst);
    node->BindBnsWithRegst(&Operator::model_diff_bns, model_diff_regst);
    node->BindBnsWithRegst(&Operator::model_bns, model_regst);
  });
}

void NormalBackwardCompTaskNode::InferBlobDescsInProducedRegsts() {
  if (GetRelatedFwTaskNode()) {
    std::shared_ptr<RegstDesc> in_diff_regst_boxing = GetProducedRegst("boxing_in_diff");
    for (std::weak_ptr<RegstDesc> regst : GetConsumedRegst("in")) {
      in_diff_regst_boxing->CopyBlobDescWithoutAddLbi(regst.lock().get());
    }

    std::shared_ptr<RegstDesc> in_diff_regst_121 = GetProducedRegst("121_in_diff");
    for (std::weak_ptr<RegstDesc> regst : GetConsumedRegst("in")) {
      in_diff_regst_121->CopyBlobDescWithoutAddLbi(regst.lock().get());
    }

    std::shared_ptr<RegstDesc> md_diff_regst = GetProducedRegst("model_diff");
    if (md_diff_regst) { md_diff_regst->CopyBlobDescFrom(GetSoleConsumedRegst("model").get()); }

    std::shared_ptr<RegstDesc> activation_diff_regst = GetProducedRegst("activation_diff");
    activation_diff_regst->CopyBlobDescWithoutAddLbi(GetSoleConsumedRegst("activation").get());
    activation_diff_regst->CopyBlobDescWithoutAddLbi(GetSoleConsumedRegst("boxing_out").get());
    activation_diff_regst->CopyBlobDescWithoutAddLbi(GetSoleConsumedRegst("121_out").get());
  } else {
    mut_exec_gph().SoleNode()->InferDiffBlobDescsWithoutFwNode(parallel_ctx());
  }
}

CompTaskNode* NormalBackwardCompTaskNode::GetRelatedFwTaskNode() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* fw_node = edge->src_node();
    if (IsForwardTaskType(fw_node->GetTaskType())) { return static_cast<CompTaskNode*>(fw_node); }
  }
  return nullptr;
}

}  // namespace oneflow
