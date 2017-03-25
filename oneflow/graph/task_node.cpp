#include "graph/task_node.h"

namespace oneflow {

// TaskNode

TaskNode::TaskNode() {
  stage_node_ = nullptr;
  related_fw_or_bp_node_ = nullptr;
  exec_graph_ = nullptr;
}

TaskNode* TaskNode::GetFwNode() const {
  CHECK(IsBpNode());
  return related_fw_or_bp_node_;
}
TaskNode* TaskNode::GetBpNode() const {
  CHECK(IsFwNode());
  return related_fw_or_bp_node_;
}

void TaskNode::set_stage_node(const StageNode* new_stage_node) {
  CHECK(IsFwNode());
  stage_node_ = new_stage_node;
}
ThreadLocalId& TaskNode::mut_thread_local_id() {
  CHECK(IsFwNode());
  return thread_local_id_;
}

std::unique_ptr<TaskNode> TaskNode::BuildAndConnectBpNode() {
  // Build
  CHECK(GetBpNode() == nullptr);
  std::unique_ptr<TaskNode> bp_node = CreateSameTypeNode();
  bp_node->InitWithFwNode(this);
  // Connect
  related_fw_or_bp_node_ = bp_node.get();
  return bp_node;
}

void TaskNode::SetNewExecGraph() {
  exec_graph_ = CreateExecGraph();
  exec_graph_->set_task_node(this);
}

std::unique_ptr<TaskNode> TaskNode::CreateSameTypeNode() const {
  LOG(FATAL) << "insignificant";
}

void TaskNode::InitWithFwNode(TaskNode* fw_node) {
  CHECK(IsBpNode());
  stage_node_ = fw_node->stage_node_;
  thread_local_id_ = fw_node->thread_local_id_;
  is_fw_node_ = false;
  related_fw_or_bp_node_ = fw_node;
}

// CopyHDTaskNode
  
const std::vector<std::string>& CopyHDTaskNode::RelatedLbns() const {
  if (IsFwInCopy()) {
    return stage_node()->chain_node()->input_lbns();
  } else {
    return stage_node()->chain_node()->output_lbns();
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
  is_fw_in_copy_ =
      of_dynamic_cast<const CopyHDTaskNode*>(fw_node)->is_fw_in_copy_;
}

// BoxingTaskNode

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
      of_dynamic_cast<const BoxingTaskNode*>(fw_node)->is_fw_in_boxing_;
}

// CompTaskNode

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
