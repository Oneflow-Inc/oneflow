#include "oneflow/core/graph/forward_compute_task_node.h"

namespace oneflow {

void ForwardCompTaskNode::ProduceAllRegstsAndBindEdges() {
  ProduceRegst("out", 1, kMaxRegisterNum);
  ProduceRegst("activation", 1, kMaxRegisterNum);
  ProduceRegst("data_tmp", 1, kMaxRegisterNum);
}

}  // namespace oneflow
