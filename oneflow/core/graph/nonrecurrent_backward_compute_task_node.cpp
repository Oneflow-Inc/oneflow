#include "oneflow/core/graph/nonrecurrent_backward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void NonRecurrentBackwardCompTaskNode::VirtualConsumeInRegst() {
  TaskNode* fw_node = GetRelatedFwTaskNode();
  for (TaskEdge* edge : fw_node->in_edges()) {
    TaskNode* pred_fw_node = edge->src_node();
    if (pred_fw_node->GetTaskType() != TaskType::kMdUpdt) {
      ConsumeRegst("in", edge->GetSoleRegst());
      return;
    }
  }
  UNEXPECTED_RUN();
}

}  // namespace oneflow
