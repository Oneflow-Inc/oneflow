#include "graph/task_node.h"

namespace oneflow {

std::unique_ptr<TaskNode> TaskNode::BuildAndConnectBpNode() {
  // Build
  CHECK(GetBpNode() == nullptr);
  std::unique_ptr<TaskNode> bp_node = CreateSameTypeNode();
  bp_node->SetupWithFwNode(this);
  // Connect
  related_fw_or_bp_node_ = bp_node.get();
  return bp_node;
}

std::unique_ptr<TaskNode> TaskNode::CreateSameTypeNode() const {
  LOG(FATAL) << "insignificant";
}

void TaskNode::SetupWithFwNode(TaskNode* fw_node) {
  CHECK(IsBpNode());
  stage_node_ = fw_node->stage_node_;
  thread_local_id_ = fw_node->thread_local_id_;
  is_fw_node_ = false;
  related_fw_or_bp_node_ = fw_node;
}

bool CompTaskNode::HasOpWithOutDiff() const {
  for (auto op : stage_node()->chain_node()->op_vec()) {
    if (! op->output_diff_blob_names().empty()) {
      return true;
    }
  }
  return false;
}

bool CompTaskNode::HasOpWithIndiff() const {
  for (auto op : stage_node()->chain_node()->op_vec()) {
    if (! op->input_diff_blob_names().empty()) {
      return true;
    }
  }
  return false;
}

} // namespace oneflow
