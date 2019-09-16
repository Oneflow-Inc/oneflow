#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/graph/sink_compute_task_node.h"

namespace oneflow {

void SinkCompTaskNode::ProduceAllRegstsAndBindEdges() {}

void SinkCompTaskNode::ConsumeAllRegsts() {
  ForEachInDataEdge([&](TaskEdge* edge) { ConsumeRegst("in", edge->GetSoleRegst()); });
}

void SinkCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();
  for (const std::string& ibn : node->op()->input_bns()) {
    node->BindBnWithOneOfTheRegsts(ibn, GetConsumedRegst("in"));
  }
  CHECK(node->op()->data_tmp_bns().empty());
  CHECK(node->op()->output_bns().empty());
}

void SinkCompTaskNode::GenerateNonCtrlRegstHandlerProto(TaskProto* task_proto) const {
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
