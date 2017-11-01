#include "oneflow/core/graph/loss_compute_task_node.h"

namespace oneflow {

void LossCompTaskNode::ProduceAllRegstsAndBindEdges() {
  NewProducedRegst("loss", 1, kMaxRegisterNum);
  NewProducedRegst("in_diff", 1, kMaxRegisterNum);
}

}  // namespace oneflow
