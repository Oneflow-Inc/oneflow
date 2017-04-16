#include "graph/copy_hd_task_node.h"
#include "operator/operator_factory.h"
#include "operator/copy_op.h"

namespace oneflow {

void CopyHDTaskNode::SetFwInCopy() {
  CHECK(IsFwNode());
  is_fw_in_copy_ = true;
}

void CopyHDTaskNode::SetFwOutCopy() {
  CHECK(IsFwNode());
  is_fw_in_copy_ = false;
}

const std::vector<std::string>& CopyHDTaskNode::CopiedLbns() const {
  return IsFwInCopy() ? chain_node()->input_lbns() : chain_node()->output_lbns();
}

void CopyHDTaskNode::InitWithFwNode(TaskNode* fw_node) {
  TaskNode::InitWithFwNode(fw_node);
  is_fw_in_copy_ = of_dynamic_cast<CopyHDTaskNode*>(fw_node)->is_fw_in_copy_;
}

} // namespace oneflow
