#include "oneflow/core/graph/repeat_forward_logical_node.h"
#include "oneflow/core/graph/repeat_backward_logical_node.h"
#include "oneflow/core/graph/repeat_forward_compute_task_node.h"

namespace oneflow {

RepeatForwardLogicalNode::RepeatForwardLogicalNode() { area_id_ = NewAreaId(); }

BackwardLogicalNode* RepeatForwardLogicalNode::NewCorrectBackwardNode() {
  return new RepeatBackwardLogicalNode();
}

CompTaskNode* RepeatForwardLogicalNode::NewCompTaskNode() const {
  return new RepeatForwardCompTaskNode();
}

int64_t RepeatForwardLogicalNode::GetAreaId() const { return area_id_; };

}  // namespace oneflow
