#include "oneflow/core/graph/tick_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void TickCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false, 1, 1);
  ForEachOutDataEdge([&](TaskEdge* edge) { BindEdgeWithProducedRegst(edge, "out"); });
}

void TickCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in");
  ForEachInDataEdge([&](TaskEdge* edge) { ConsumeRegst("in", edge->GetSoleRegst()); });
}

void TickCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();
  const std::list<std::shared_ptr<RegstDesc>>& in_regsts = GetConsumedRegst("in");
  for (const std::string& ibn : node->op()->input_bns()) {
    node->BindBnWithOneOfTheRegsts(ibn, in_regsts);
  }
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  for (const std::string& obn : node->op()->output_bns()) {
    const LogicalBlobId& lbi = node->op()->BnInOp2Lbi(obn);
    out_regst->AddLbi(lbi);
    node->BindBnWithRegst(obn, out_regst);
  }
  node->InferBlobDescs(parallel_ctx());
}

void TickCompTaskNode::InferProducedDataRegstTimeShape() {
  auto in_time_shape = (*in_edges().begin())->src_node()->GetFastestInputOutputTimeShape();
  for (TaskEdge* edge : in_edges()) {
    CHECK(*in_time_shape == *edge->src_node()->GetFastestInputOutputTimeShape());
  }
  *GetProducedRegst("out")->mut_data_regst_time_shape() = in_time_shape;
}

}  // namespace oneflow
