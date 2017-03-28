#include "graph/task_node.h"

namespace oneflow {

TaskNode::TaskNode() {
  stage_node_ = nullptr;
  related_fw_or_bp_node_ = nullptr;
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

void TaskNode::BuildExecGraphAndSetRegisterDescs() {
  if (IsFwNode()) {
    FwBuildExecGraphAndSetProducedRegisterDescs();
  } else {
    BpBuildExecGraphAndSetProducedRegisterDescs();
  }
  SubscribeRegisterDescInnerPath();
}

std::unique_ptr<TaskNode> TaskNode::CreateSameTypeNode() const {
  LOG(FATAL) << "insignificant";
}

void TaskNode::InitWithFwNode(TaskNode* fw_node) {
  stage_node_ = fw_node->stage_node_;
  thread_local_id_ = fw_node->thread_local_id_;
  is_fw_node_ = false;
  related_fw_or_bp_node_ = fw_node;
}

void TaskNode::SubscribeRegisterDescInnerPath() {
  for (const TaskEdge* edge : in_edges()) {
    edge->register_desc()->AddSubscriber(this);
    subscribed_register_descs_.insert(edge->register_desc());
  }
}

} // namespace oneflow
