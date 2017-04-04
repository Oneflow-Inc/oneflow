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

void TaskNode::BuildExecGraphAndSetRegisters(Path* path) {
  if (IsFwNode()) {
    FwBuildExecGraphAndSetProducedRegisters(path);
  } else {
    BpBuildExecGraphAndSetProducedRegisters(path);
  }
  SubscribeRegisterDescInnerPath();
}

std::unique_ptr<TaskNode> TaskNode::CreateSameTypeNode() const {
  UNEXPECTED_RUN();
}

void TaskNode::InitWithFwNode(TaskNode* fw_node) {
  stage_node_ = fw_node->stage_node_;
  thread_local_id_ = fw_node->thread_local_id_;
  is_fw_node_ = false;
  related_fw_or_bp_node_ = fw_node;
}

void TaskNode::BindProducedRegisterAndOutEdge(RegisterDesc* regi,
                                              const TaskEdge* edge) {
  CHECK(produced_register2out_edge.emplace(regi, edge).second);
  CHECK(out_edge2produced_register.emplace(edge, regi).second);
}

const TaskEdge* TaskNode::GetOutEdge4ProducedRegister(RegisterDesc* regi) const {
  return produced_register2out_edge.at(regi);
}

RegisterDesc* TaskNode::GetProducedRegister4OutEdge(const TaskEdge* edge) const {
  return out_edge2produced_register.at(edge);
}


void TaskNode::SubscribeRegisterDescInnerPath() {
  for (const TaskEdge* edge : in_edges()) {
    RegisterDesc* regi =  GetRelatedRegister(edge);
    regi->AddSubscriber(this);
    subscribed_register_descs_.insert(regi);
  }
}

void TaskNode::AddInPathLbn2ProducedRegister() {
  for (const std::unique_ptr<ExecNode>& node : exec_graph_.nodes()) {
    for (const auto& pair : node->produced_lbn_regi_pairs()) {
      const std::string& lbn = pair.first;
      RegisterDesc* register_desc = pair.second;
      register_desc->AddLbn(lbn);
    }
  }
}

} // namespace oneflow
