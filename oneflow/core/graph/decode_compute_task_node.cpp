#include "oneflow/core/graph/decode_compute_task_node.h"
#include "oneflow/core/graph/task_graph.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void DecodeCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("data_tmp", true, 1, 1);
  ProduceRegst("out", true);
  ProduceRegst("fw_pb_out", false);
  for (TaskEdge* edge : out_edges()) {
    BindEdgeWithProducedRegst(edge, "out");
    if (edge->dst_node()->GetTaskType() != TaskType::kCopyHd) {
      BindEdgeWithProducedRegst(edge, "fw_pb_out");
    }
  }
}

void DecodeCompTaskNode::ConsumeAllRegsts() {
  if (in_edges().size() == 1) {
    ConsumeRegst("record", SoleInEdge()->GetSoleRegst());
  } else {
    CHECK_EQ(in_edges().size(), 0);
  }
}

void DecodeCompTaskNode::BuildExecGphAndRegst() {
  std::shared_ptr<RegstDesc> data_tmp_regst = GetProducedRegst("data_tmp");
  std::shared_ptr<RegstDesc> out_regst = GetProducedRegst("out");
  std::shared_ptr<RegstDesc> fw_pb_out_regst = GetProducedRegst("fw_pb_out");
  std::shared_ptr<RegstDesc> record_regst = GetSoleConsumedRegst("record");
  ExecNode* node = mut_exec_gph().NewNode();
  node->mut_op() = logical_node()->SoleOp();
  node->BindBnWithRegst(node->op()->SoleIbn(), record_regst);
  node->AddBnToRegstAndBindIt(&Operator::output_bns, out_regst);
  node->AddBnToRegstAndBindIt(&Operator::pb_output_bns, fw_pb_out_regst);
  node->AddBnToRegstAndBindIt(&Operator::data_tmp_bns, data_tmp_regst);
  node->InferBlobDescs(parallel_ctx());
}

void DecodeCompTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
