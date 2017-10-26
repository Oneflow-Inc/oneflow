#include "oneflow/core/graph/source_compute_task_node.h"

namespace oneflow {

void SourceCompTaskNode::NewAllProducedRegst() {
  NewProducedRegst("out", 1, kMaxRegisterNum);
}

}  // namespace oneflow
