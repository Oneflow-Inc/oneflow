#include "oneflow/core/graph/nccl_inter_device_reduce_backward_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void NcclInterDeviceReduceBackwardCompTaskNode::ConsumeAllRegsts() {
  for (const TaskEdge* edge : in_edges()) {
    if (edge->src_node()->GetTaskType() == TaskType::kNcclInterDeviceReduceForward) { continue; }
    CHECK(consumed_regsts().empty());
    ConsumeRegst("out_diff", edge->GetSoleRegst());
  }
}

void NcclInterDeviceReduceBackwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("bw_buf", true, 1, 1);
  std::shared_ptr<RegstDesc> in_diff_regst = ProduceRegst("in_diff", true);
  SoleOutEdge()->AddRegst("in_diff", in_diff_regst);
}

void NcclInterDeviceReduceBackwardCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> sole_op = this->logical_node()->SoleOp();
  node->mut_op() = sole_op;
  node->BindBnWithRegst(sole_op->SoleOdbn(), GetSoleConsumedRegst("out_diff"));
  node->AddBnToRegstAndBindIt(&Operator::input_diff_bns, GetProducedRegst("in_diff"));
  node->AddBnToRegstAndBindIt(&Operator::bw_buf_bns, GetProducedRegst("bw_buf"));
  node->InferDiffBlobDescsWithoutFwNode(parallel_ctx());
  node->InferBwBufBlobDescs(parallel_ctx());
}

void NcclInterDeviceReduceBackwardCompTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
