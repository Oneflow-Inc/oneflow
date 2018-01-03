#include "oneflow/core/graph/recurrent_backward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void RecurrentBackwardCompTaskNode::VirtualConsumeInRegst() {
  CompTaskNode* fw_node = static_cast<CompTaskNode*>(GetRelatedFwTaskNode());
  std::shared_ptr<const Operator> op = fw_node->chain_node()->SoleOp();
  for (TaskEdge* edge : fw_node->in_edges()) {
    TaskNode* pred_fw_node = edge->src_node();
    if (pred_fw_node->GetTaskType() == TaskType::kMdUpdt) { continue; }
    std::shared_ptr<RegstDesc> regst = edge->GetSoleRegst();
    if (regst->GetBlobDesc(op->Lbn4BnInOp("in"))) {
      ConsumeRegst("in", regst);
    } else if (regst->GetBlobDesc(op->Lbn4BnInOp("h0"))) {
      ConsumeRegst("h0", regst);
    } else {
      UNEXPECTED_RUN();
    }
  }
}

}  // namespace oneflow
