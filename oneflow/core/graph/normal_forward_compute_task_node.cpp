#include "oneflow/core/graph/normal_forward_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void NormalForwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("boxing_out");
  ProduceRegst("121_out");
  ProduceRegst("activation");
  ProduceRegst("data_tmp");
  ProduceRegst("forward_model");
  for (TaskEdge* edge : out_edges()) {
    const LogicalNode* succ_logical = GetOneSuccLogicalNodeOnEdge(edge);
    if (succ_logical->TypeName() == "MdSave") {
      BindEdgeWithProducedRegst(edge, "forward_model");
    } else if (succ_logical->TypeName() == "NormalBackward") {
      BindEdgeWithProducedRegst(edge, "boxing_out");
      BindEdgeWithProducedRegst(edge, "121_out");
      BindEdgeWithProducedRegst(edge, "activation");
      BindEdgeWithProducedRegst(edge, "data_tmp");
    } else {
      BldSubTskGphMthd mthd = GetMthdForBldSubTskGph(logical_node(), succ_logical);
      if (mthd == &TaskGraph::BldSubTskGphByBoxing) {
        BindEdgeWithProducedRegst(edge, "boxing_out");
      } else if (mthd == &TaskGraph::BldSubTskGphByOneToOne) {
        BindEdgeWithProducedRegst(edge, "121_out");
      } else {
        UNIMPLEMENTED();
      }
    }
  }
}

void NormalForwardCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    const LogicalNode* pred_logical = GetOnePredLogicalNodeOnEdge(edge);
    if (pred_logical->TypeName() == "NormalMdUpdt") {
      ConsumeRegst("model", edge->GetRegst("model"));
      ConsumeRegst("model_tmp", edge->GetRegst("model_tmp"));
    } else {
      ConsumeRegst("in", edge->GetSoleRegst());
    }
  }
}

bool NormalForwardCompTaskNode::IsReadyForBuild() {
  for (std::weak_ptr<RegstDesc> regst_desc : GetConsumedRegst("in")) {
    if (regst_desc.lock()->IsLocked() == false) { return false; }
  }
  return true;
}

void NormalForwardCompTaskNode::ToProto(TaskProto* task_proto) {
  CompTaskNode::ToProto(task_proto);
  task_proto->set_random_seed(random_seed_);
}

void NormalForwardCompTaskNode::BuildExecGphAndRegst() {
  BuildExecGphStructAndBindInRegst();
  BuildOutRegst();
  BuildActivationRegst();
  BuildModelAndTmpRegsts();
  BuildForwardModelRegsts();
  mut_exec_gph().TopoForEachNode([this](ExecNode* node) { node->InferBlobDescs(parallel_ctx()); });
}

void NormalForwardCompTaskNode::LockRegsts() {
  TaskNode::LockRegsts();
  TryLockConsumedRegst("model");
  TryLockConsumedRegst("model_tmp");
}

void NormalForwardCompTaskNode::BuildExecGphStructAndBindInRegst() {
  HashMap<LogicalBlobId, std::pair<ExecNode*, std::string>> lbi2producer;
  for (std::shared_ptr<const Operator> op : logical_node()->op_vec()) {
    ExecNode* cur_node = mut_exec_gph().NewNode();
    cur_node->mut_op() = op;
    for (const std::string& obn : op->output_bns()) {
      const LogicalBlobId& lbi = op->BnInOp2Lbi(obn);
      CHECK(lbi2producer.insert({lbi, {cur_node, obn}}).second);
    }
  }
  const std::list<std::weak_ptr<RegstDesc>>& in_regsts = GetConsumedRegst("in");
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    for (const std::string& ibn : cur_node->op()->input_bns()) {
      const LogicalBlobId& lbi = cur_node->op()->BnInOp2Lbi(ibn);
      auto producer_it = lbi2producer.find(lbi);
      if (producer_it != lbi2producer.end()) {
        ExecEdge* edge = mut_exec_gph().NewEdge();
        edge->set_lbi(lbi);
        edge->mut_src_bn() = producer_it->second.second;
        edge->mut_dst_bn() = ibn;
        Connect(producer_it->second.first, edge, cur_node);
      } else {
        cur_node->BindBnWithOneOfTheRegsts(ibn, in_regsts);
      }
    }
  });
}

void NormalForwardCompTaskNode::BuildOutRegst() {
  std::shared_ptr<RegstDesc> out_regst_121 = GetProducedRegst("121_out");
  std::shared_ptr<RegstDesc> out_regst_boxing = GetProducedRegst("boxing_out");
  const HashSet<LogicalBlobId>& lbi_boxing = logical_node()->lbi_boxing();
  const HashSet<LogicalBlobId>& lbi_121 = logical_node()->lbi_121();
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    HashSet<LogicalBlobId> found_lbis;
    for (ExecEdge* out_edge : cur_node->out_edges()) { found_lbis.insert(out_edge->lbi()); }
    for (const std::string& obn : cur_node->op()->output_bns()) {
      const LogicalBlobId& lbi = cur_node->op()->BnInOp2Lbi(obn);
      if (found_lbis.find(lbi) != found_lbis.end()) {
        CHECK(lbi_boxing.find(lbi) == lbi_boxing.end());
        CHECK(lbi_121.find(lbi) == lbi_121.end());
        continue;
      }
      if (lbi_boxing.find(lbi) != lbi_boxing.end()) {
        out_regst_boxing->AddLbi(lbi);
        cur_node->BindBnWithRegst(obn, out_regst_boxing);
      } else if (lbi_121.find(lbi) != lbi_121.end()) {
        out_regst_121->AddLbi(lbi);
        cur_node->BindBnWithRegst(obn, out_regst_121);
      } else {
        UNIMPLEMENTED();
      }
    }
  });
}

void NormalForwardCompTaskNode::BuildActivationRegst() {
  std::shared_ptr<RegstDesc> activation_regst = GetProducedRegst("activation");
  mut_exec_gph().ForEachEdge([&](const ExecEdge* edge) {
    if (activation_regst->GetBlobDesc(edge->lbi()) == nullptr) {
      activation_regst->AddLbi(edge->lbi());
      edge->src_node()->BindBnWithRegst(edge->src_bn(), activation_regst);
    }
    edge->dst_node()->BindBnWithRegst(edge->dst_bn(), activation_regst);
  });
}

void NormalForwardCompTaskNode::BuildModelAndTmpRegsts() {
  std::shared_ptr<RegstDesc> model_regst = GetSoleConsumedRegst("model");
  std::shared_ptr<RegstDesc> model_tmp_regst = GetSoleConsumedRegst("model_tmp");
  mut_exec_gph().ForEachNode([&](ExecNode* node) {
    node->AddBnToRegstAndBindIt(&Operator::data_tmp_bns, GetProducedRegst("data_tmp"));
    for (const std::string& mtbn : node->op()->model_tmp_bns()) {
      if (!model_tmp_regst->IsLocked()) {
        const LogicalBlobId& lbi = node->op()->BnInOp2Lbi(mtbn);
        model_tmp_regst->AddLbi(lbi);
      }
      node->BindBnWithRegst(mtbn, model_tmp_regst);
    }
    for (const std::string& mbn : node->op()->model_bns()) {
      if (!model_regst->IsLocked()) {
        const LogicalBlobId& lbi = node->op()->BnInOp2Lbi(mbn);
        model_regst->AddLbi(lbi);
      }
      node->BindBnWithRegst(mbn, model_regst);
    }
  });
}

void NormalForwardCompTaskNode::BuildForwardModelRegsts() {
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    cur_node->AddBnToRegstAndBindIt(&Operator::forward_model_bns,
                                    GetProducedRegst("forward_model"));
  });
}

}  // namespace oneflow
