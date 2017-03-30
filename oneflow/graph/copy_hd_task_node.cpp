#include "graph/copy_hd_task_node.h"

namespace oneflow {

const std::vector<std::string>& CopyHDTaskNode::CopiedLbns() const {
  if (IsFwInCopy()) {
    return chain_node()->input_lbns();
  } else {
    return chain_node()->output_lbns();
  }
}

void CopyHDTaskNode::SetFwInCopy() {
  CHECK(IsFwNode());
  is_fw_in_copy_ = true;
}

void CopyHDTaskNode::SetFwOutCopy() {
  CHECK(IsFwNode());
  is_fw_in_copy_ = false;
}

void CopyHDTaskNode::InitWithFwNode(TaskNode* fw_node) {
  TaskNode::InitWithFwNode(fw_node);
  is_fw_in_copy_ = of_dynamic_cast<CopyHDTaskNode*>(fw_node)->is_fw_in_copy_;
}

} // namespace oneflow
