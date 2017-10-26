#include "oneflow/core/graph/forward_compute_task_node.h"

namespace oneflow {

void FwCompTaskNode::NewAllProducedRegst() {
  NewProducedRegst("out", 1, kMaxRegisterNum);
  NewProducedRegst("activation", 1, kMaxRegisterNum);
  NewProducedRegst("data_tmp", 1, kMaxRegisterNum);
}

}  // namespace oneflow
