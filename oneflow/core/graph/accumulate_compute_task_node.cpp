#include "oneflow/core/graph/loss_accumulate_compute_task_node.h"

namespace oneflow {

void AccCompTaskNode::ProduceAllRegstsAndBindEdges() {
  NewProducedRegst("acc", 1, kMaxRegisterNum);
}

}  // namespace oneflow
