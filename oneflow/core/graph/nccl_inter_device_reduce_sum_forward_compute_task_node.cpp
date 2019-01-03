#include "oneflow/core/graph/nccl_inter_device_reduce_sum_forward_compute_task_node.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

void NcclInterDeviceReduceSumForwardCompTaskNode::ConsumeAllRegsts() {
  ConsumeRegst("in", SoleInEdge()->GetSoleRegst());
}

void NcclInterDeviceReduceSumForwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("fw_buf", true, 1, 1);
  std::shared_ptr<RegstDesc> out_regst = ProduceRegst("out", true);
  for (TaskEdge* edge : out_edges()) {
    if (edge->dst_node()->GetTaskType() == TaskType::kNcclInterDeviceReduceSumBackward) {
      continue;
    }
    edge->AddRegst("out", out_regst);
  }
}

void NcclInterDeviceReduceSumForwardCompTaskNode::BuildExecGphAndRegst() {
  ExecNode* node = mut_exec_gph().NewNode();
  std::shared_ptr<Operator> sole_op = this->logical_node()->SoleOp();
  node->mut_op() = sole_op;
  node->BindBnWithRegst(sole_op->SoleIbn(), GetSoleConsumedRegst("in"));
  node->AddBnToRegstAndBindIt(&Operator::output_bns, GetProducedRegst("out"));
  node->AddBnToRegstAndBindIt(&Operator::fw_buf_bns, GetProducedRegst("fw_buf"));
  node->InferBlobDescs(parallel_ctx());
}

void NcclInterDeviceReduceSumForwardCompTaskNode::InferProducedDataRegstTimeShape() {
  NaiveInferProducedDataRegstTimeShape();
}

}  // namespace oneflow
