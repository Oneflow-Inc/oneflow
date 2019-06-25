#include "oneflow/core/graph/decode_random_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void DecodeRandomCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst());
}

void DecodeRandomCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false);
  ForEachOutDataEdge([&](TaskEdge* edge) { BindEdgeWithProducedRegst(edge, "out"); });
}

void DecodeRandomCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();
  node->BindBnWithRegst(node->op()->SoleIbn(), GetSoleConsumedRegst("in"));
  node->AddBnToRegstAndBindIt(&Operator::output_bns, out_regst);
  node->InferBlobDescs(parallel_ctx());
}

void DecodeRandomCompTaskNode::InferProducedDataRegstTimeShape() {
  std::shared_ptr<Shape> time_shape(
      new Shape({GlobalJobDesc().TotalBatchNum(), GlobalJobDesc().NumOfPiecesInBatch()}));

  ForEachProducedDataRegst([time_shape](const std::string& name, RegstDesc* regst) {
    *regst->mut_data_regst_time_shape() = time_shape;
  });
}

}  // namespace oneflow
