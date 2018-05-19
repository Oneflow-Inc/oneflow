#include "oneflow/core/graph/normal_forward_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void NormalForwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceB121Regst("out");
  ProduceRegst("activation");
  ProduceRegst("data_tmp");
  ProduceRegst("forward_model");
  ProduceRegst("const_buf", 1, 1);
  for (TaskEdge* edge : out_edges()) {
    const LogicalNode* succ_logical = GetOneSuccLogicalNodeOnEdge(edge);
    if (succ_logical->TypeName() == "MdSave") {
      BindEdgeWithProducedRegst(edge, "forward_model");
    } else if (succ_logical->TypeName() == "NormalBackward") {
      BindEdgeWithProducedRegst(edge, "boxing_out");
      BindEdgeWithProducedRegst(edge, "121_out");
      BindEdgeWithProducedRegst(edge, "activation");
      BindEdgeWithProducedRegst(edge, "data_tmp");
      BindEdgeWithProducedRegst(edge, "const_buf");
    } else {
      BindEdgeWithProducedB121Regst(edge, "out");
    }
  }
}

void NormalForwardCompTaskNode::ConsumeAllRegsts() {
  for (TaskEdge* edge : in_edges()) {
    TaskNode* src_node = edge->src_node();
    TaskType src_task_type = src_node->GetTaskType();
    if (src_task_type == TaskType::kNormalMdUpdt) {
      ConsumeRegst("model", edge->GetRegst("model"));
      ConsumeRegst("const_model", edge->GetRegst("const_model"));
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
  BuildModel7ConstModel7DataTmp7BufRegsts();
  BuildForwardModelRegsts();
  mut_exec_gph().TopoForEachNode([this](ExecNode* node) { node->InferBlobDescs(parallel_ctx()); });
}

void NormalForwardCompTaskNode::LockRegsts() {
  TaskNode::LockRegsts();
  TryLockConsumedRegst("model");
  TryLockConsumedRegst("const_model");
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
  mut_exec_gph().ForEachNode([&](ExecNode* cur_node) {
    HashSet<LogicalBlobId> found_lbis;
    for (ExecEdge* out_edge : cur_node->out_edges()) { found_lbis.insert(out_edge->lbi()); }
    for (const std::string& obn : cur_node->op()->output_bns()) {
      const LogicalBlobId& lbi = cur_node->op()->BnInOp2Lbi(obn);
      if (TryAddLbiToB121RegstAndBindIt(cur_node, obn, "out") == false) {
        CHECK(found_lbis.find(lbi) != found_lbis.end());
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

void NormalForwardCompTaskNode::BuildModel7ConstModel7DataTmp7BufRegsts() {
  std::shared_ptr<RegstDesc> model_regst = GetSoleConsumedRegst("model");
  std::shared_ptr<RegstDesc> const_model_regst = GetSoleConsumedRegst("const_model");
  mut_exec_gph().ForEachNode([&](ExecNode* node) {
    node->AddBnToRegstAndBindIt(&Operator::data_tmp_bns, GetProducedRegst("data_tmp"));
    node->AddBnToRegstAndBindIt(&Operator::const_buf_bns, GetProducedRegst("const_buf"));
    for (const std::string& cmbn : node->op()->const_model_bns()) {
      if (!const_model_regst->IsLocked()) {
        const LogicalBlobId& lbi = node->op()->BnInOp2Lbi(cmbn);
        const_model_regst->AddLbi(lbi);
      }
      node->BindBnWithRegst(cmbn, const_model_regst);
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
