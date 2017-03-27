#include "graph/boxing_task_node.h"

namespace oneflow {

void BoxingTaskNode::SetFwInBoxing() {
  CHECK(IsFwNode());
  is_fw_in_boxing_ = true;
}

void BoxingTaskNode::SetFwOutBoxing() {
  CHECK(IsFwNode());
  is_fw_in_boxing_ = false;
}

void BoxingTaskNode::InitWithFwNode(TaskNode* fw_node) {
  TaskNode::InitWithFwNode(fw_node);
  is_fw_in_boxing_ =
      of_dynamic_cast<BoxingTaskNode*>(fw_node)->is_fw_in_boxing_;
}

} // namespace oneflow
