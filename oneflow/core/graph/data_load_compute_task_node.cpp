#include "oneflow/core/graph/data_load_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void DataLoadCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInDataEdge()->GetSoleRegst());
}

void DataLoadCompTaskNode::ProduceAllRegstsAndBindEdges() {
  // ProduceRegst("out", false);
  ProduceRegst("out", false, 2, 2);
  ForEachOutDataEdge([&](TaskEdge* edge) { BindEdgeWithProducedRegst(edge, "out"); });
  // std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", false, 2, 2);
  // ForEachOutDataEdge([&](TaskEdge* edge) { edge->AddRegst("out", out_regst); });
}

void DataLoadCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();
  node->BindBnWithRegst(node->op()->SoleIbn(), GetSoleConsumedRegst("in"));
  node->AddBnToRegstAndBindIt(&Operator::output_bns, out_regst);
  node->InferBlobDescs(parallel_ctx());
}

void DataLoadCompTaskNode::InferProducedDataRegstTimeShape() {
  std::shared_ptr<Shape> time_shape(
      new Shape({GlobalJobDesc().TotalBatchNum(), GlobalJobDesc().NumOfPiecesInBatch()}));

  ForEachProducedDataRegst([time_shape](const std::string& name, RegstDesc* regst) {
    *regst->mut_data_regst_time_shape() = time_shape;
  });
}

}  // namespace oneflow
