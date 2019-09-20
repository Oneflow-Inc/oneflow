#include "oneflow/core/graph/loss_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void LossCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", true);
  ProduceRegst("tmp", true, 1, 1);
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
  std::shared_ptr<RegstDesc> tmp_regst = GetProducedRegst("tmp");
  loss_node->AddBnToRegstAndBindIt(&Operator::tmp_bns, tmp_regst);
  loss_node->AddBnToRegstAndBindIt(&Operator::const_buf_bns, GetProducedRegst("const_buf"));

  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  for (const std::string& obn : loss_op->output_bns()) {
    out_regst->AddLbi(loss_op->BnInOp2Lbi(obn));
    loss_node->BindBnWithRegst(obn, out_regst);
  }
  mut_exec_gph().TopoForEachNode([this](ExecNode* node) { node->InferBlobDescs(parallel_ctx()); });
}

void LossCompTaskNode::InferProducedDataRegstTimeShape() { NaiveInferProducedDataRegstTimeShape(); }

}  // namespace oneflow
