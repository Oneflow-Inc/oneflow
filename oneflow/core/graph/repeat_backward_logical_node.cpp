#include "oneflow/core/graph/repeat_backward_logical_node.h"
#include "oneflow/core/graph/repeat_backward_compute_task_node.h"

namespace oneflow {

RepeatBackwardLogicalNode::RepeatBackwardLogicalNode() { area_id_ = NewAreaId(); }

CompTaskNode* RepeatBackwardLogicalNode::NewCompTaskNode() const {
  return new RepeatBackwardCompTaskNode();
}

int64_t RepeatBackwardLogicalNode::GetAreaId() const { return area_id_; };

}  // namespace oneflow