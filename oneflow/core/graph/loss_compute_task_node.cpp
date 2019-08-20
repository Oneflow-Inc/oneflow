#include "oneflow/core/graph/loss_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void LossCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", true);
  ProduceRegst("data_tmp", true, 1, 1);
  ProduceRegst("const_buf", false, 1, 1);
  ForEachOutDataEdge([&](TaskEdge* edge) { BindEdgeWithProducedRegst(edge, "out"); });
}

void LossCompTaskNode::ConsumeAllRegsts() {
  ForEachInDataEdge([&](TaskEdge* edge) { ConsumeRegst("in", edge->GetSoleRegst()); });
}

void LossCompTaskNode::BuildExecGphAndRegst() {
  const auto& op_vec = logical_node()->op_vec();
  CHECK_EQ(op_vec.size(), 1);
  std::shared_ptr<const Operator> loss_op = op_vec[0];
  ExecNode* loss_node = mut_exec_gph().NewNode();
  loss_node->mut_op() = loss_op;
  for (const std::string& ibn : loss_op->input_bns()) {
    loss_node->BindBnWithOneOfTheRegsts(ibn, GetConsumedRegst("in"));
  }
  std::shared_ptr<RegstDesc> data_tmp_regst = GetProducedRegst("data_tmp");
  loss_node->AddBnToRegstAndBindIt(&Operator::data_tmp_bns, data_tmp_regst);
  loss_node->AddBnToRegstAndBindIt(&Operator::const_buf_bns, GetProducedRegst("const_buf"));

  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  for (const std::string& obn : loss_op->output_bns()) {
    out_regst->AddLbi(loss_op->BnInOp2Lbi(obn));
    loss_node->BindBnWithRegst(obn, out_regst);
  }
  mut_exec_gph().TopoForEachNode([this](ExecNode* node) { node->InferBlobDescs(parallel_ctx()); });
  mut_exec_gph().TopoForEachNode(
      [this](ExecNode* node) { node->FixInDiffBlobDescs(parallel_ctx()); });
}

void LossCompTaskNode::InferProducedDataRegstTimeShape() { NaiveInferProducedDataRegstTimeShape(); }

void LossCompTaskNode::GenerateProto4Actor(TaskProto* task_proto) {
  HashSet<int64_t> ctrl_ids = GetAllCtrlRegstDescIds();
  RegstHandlerProto* ctrl_handler_proto = nullptr;
  if (!ctrl_ids.empty()) {
    ctrl_handler_proto = task_proto->mutable_regst_handlers()->Add();
    ctrl_handler_proto->set_type("Ctrl");
  }
  RegstHandlerProto* naive_handler_proto = task_proto->mutable_regst_handlers()->Add();
  naive_handler_proto->set_type("Naive");

  for (const auto& pair : consumed_regsts()) {
    for (const auto& regst_desc : pair.second) {
      int64_t regst_desc_id = regst_desc->regst_desc_id();
      if (IsKeyFound(ctrl_ids, regst_desc_id)) {
        ctrl_handler_proto->mutable_consumed_regst_desc_ids()->add_regst_desc_id(regst_desc_id);
      } else {
        naive_handler_proto->mutable_consumed_regst_desc_ids()->add_regst_desc_id(regst_desc_id);
      }
    }
  }
  for (const auto& pair : produced_regsts()) {
    int64_t regst_desc_id = pair.second->regst_desc_id();
    if (IsKeyFound(ctrl_ids, regst_desc_id)) {
      ctrl_handler_proto->mutable_produced_regst_desc_ids()->add_regst_desc_id(regst_desc_id);
    } else {
      naive_handler_proto->mutable_produced_regst_desc_ids()->add_regst_desc_id(regst_desc_id);
    }
  }
}

}  // namespace oneflow
