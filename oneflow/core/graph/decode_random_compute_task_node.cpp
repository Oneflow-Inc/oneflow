#include "oneflow/core/graph/decode_random_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"
#include "decode_random_compute_task_node.h"

namespace oneflow {

void DecodeRandomCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", true);
  for (TaskEdge* edge : out_edges()) { BindEdgeWithProducedRegst(edge, "out"); }
}

void DecodeRandomCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();
  node->AddBnToRegstAndBindIt(&Operator::output_bns, out_regst);
  node->InferBlobDescs(parallel_ctx());
}

void DecodeRandomCompTaskNode::InferProducedRegstTimeShape() {
  std::shared_ptr<Shape> time_shape;
  time_shape.reset(new Shape(
      {Global<JobDesc>::Get()->TotalBatchNum(), Global<JobDesc>::Get()->NumOfPiecesInBatch()}));
  for (auto& pair : produced_regsts()) { pair.second->mut_time_shape() = time_shape; }
}

}  // namespace oneflow
