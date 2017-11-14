#include "oneflow/core/graph/loss_compute_task_node.h"

namespace oneflow {

void LossCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("loss", 1, kMaxRegisterNum);
  ProduceRegst("in_diff", 1, kMaxRegisterNum);
}

}  // namespace oneflow
