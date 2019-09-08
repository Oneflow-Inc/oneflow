#include "oneflow/core/graph/decode_random_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void DecodeRandomCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", false);
  ForEachOutDataEdge([&](TaskEdge* edge) { BindEdgeWithProducedRegst(edge, "out"); });
}

void DecodeRandomCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();
  node->AddBnToRegstAndBindIt(&Operator::output_bns, out_regst);
  node->InferBlobDescs(parallel_ctx());
}

void DecodeRandomCompTaskNode::InferProducedDataRegstTimeShape() {
  std::shared_ptr<Shape> time_shape(new Shape(
      {Global<JobDesc>::Get()->TotalBatchNum(), Global<JobDesc>::Get()->NumOfPiecesInBatch()}));

  ForEachProducedDataRegst([time_shape](const std::string& name, RegstDesc* regst) {
    *regst->mut_data_regst_time_shape() = time_shape;
  });
}

void DecodeRandomCompTaskNode::GenerateNonCtrlRegstHandlerProto(TaskProto* task_proto) const {
  RegstHandlerProto naive_proto = CreateRegstHandlerProto("Naive");
  ForEachNonCtrlConsumedRegstDescId([&](int64_t regst_desc_id) {
    naive_proto.mutable_consumed_regst_desc_ids()->add_regst_desc_id(regst_desc_id);
  });
  ForEachNonCtrlProducedRegstDescId([&](int64_t regst_desc_id) {
    naive_proto.mutable_produced_regst_desc_ids()->add_regst_desc_id(regst_desc_id);
  });
  if (!IsRegstHandlerProtoEmpty(naive_proto)) {
    *(task_proto->mutable_regst_handlers()->Add()) = naive_proto;
  }
}

}  // namespace oneflow
