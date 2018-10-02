#include "oneflow/core/graph/repeat_forward_logical_node.h"
#include "oneflow/core/graph/repeat_backward_logical_node.h"
#include "oneflow/core/graph/repeat_forward_compute_task_node.h"

namespace oneflow {

BackwardLogicalNode* RepeatForwardLogicalNode::NewCorrectBackwardNode() {
  return new RepeatBackwardLogicalNode();
}

CompTaskNode* RepeatForwardLogicalNode::NewCompTaskNode() const {
  return new RepeatForwardCompTaskNode();
}

int64_t RepeatForwardLogicalNode::GetAreaId() const { return AreaType::kDataForwardArea; };

}  // namespace oneflow