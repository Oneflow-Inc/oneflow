#include "oneflow/core/graph/decode_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void DecodeCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("tmp", true, 1, 1);
  ProduceRegst("out", true);
  ForEachOutDataEdge([&](TaskEdge* edge) { BindEdgeWithProducedRegst(edge, "out"); });
}

void DecodeCompTaskNode::ConsumeAllRegsts() {
  if (in_data_edges_size() == 1) {
    ConsumeRegst("record", SoleInDataEdge()->GetSoleRegst());
  } else {
    CHECK_EQ(in_data_edges_size(), 0);
  }
}

void DecodeCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> tmp_regst = GetProducedRegst("tmp");
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  std::shared_ptr<RegstDesc> record_regst = GetSoleConsumedRegst("record");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();
  node->BindBnWithRegst(node->op()->SoleIbn(), record_regst);
  node->AddBnToRegstAndBindIt(&Operator::output_bns, out_regst);
  node->AddBnToRegstAndBindIt(&Operator::tmp_bns, tmp_regst);
  node->InferBlobDescs(parallel_ctx());
}

void DecodeCompTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
