#include "oneflow/core/graph/forward_compute_task_node.h"
#include "oneflow/core/graph/nonrecurrent_forward_compute_task_node.h"
#include "oneflow/core/graph/chain_node.h"

namespace oneflow {

void NonRecurrentForwardCompTaskNode::VirtualConsumeInRegst(TaskEdge* edge) {
  ConsumeRegst("in", edge->GetSoleRegst());
}

bool NonRecurrentForwardCompTaskNode::IsReadyForBuild() {
  return GetConsumedRegst("in")->IsLocked();
}

}  // namespace oneflow
