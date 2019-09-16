#include "oneflow/core/graph/decode_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void DecodeCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("data_tmp", true, 1, 1);
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
  std::shared_ptr<RegstDesc> data_tmp_regst = GetProducedRegst("data_tmp");
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  std::shared_ptr<RegstDesc> record_regst = GetSoleConsumedRegst("record");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();
  node->BindBnWithRegst(node->op()->SoleIbn(), record_regst);
  node->AddBnToRegstAndBindIt(&Operator::output_bns, out_regst);
  node->AddBnToRegstAndBindIt(&Operator::data_tmp_bns, data_tmp_regst);
  node->InferBlobDescs(parallel_ctx());
}

void DecodeCompTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

void DecodeCompTaskNode::GenerateNonCtrlRegstHandlerProto(TaskProto* task_proto) const {
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
