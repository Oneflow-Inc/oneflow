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
ThrdLocId& TaskNode::mut_thrd_loc_id() {
  CHECK(IsFwNode());
  return thrd_loc_id_;
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

void TaskNode::BuildExecAndProducedRegstsAndSubscribeInPath(Path* path) {
  SubscribeRegstDescInnerPath();
  if (IsFwNode()) {
    FwBuildExecAndProducedRegsts(path);
  } else {
    BpBuildExecAndProducedRegsts(path);
  }
}

void TaskNode::Subscribe(RegstDesc* regst) {
  regst->AddSubscriber(this);
  CHECK(subscribed_regst_descs_.insert(regst).second);
}

RegstDesc* TaskNode::GetProducedRegstDesc(const std::string& regst_desc_name) {
  return produced_regst_descs_.at(regst_desc_name).get();
}

const TaskEdge* TaskNode::GetOutEdge4ProducedRegst(RegstDesc* regst) const {
  return produced_regst2out_edge.at(regst);
}

RegstDesc* TaskNode::GetProducedRegst4OutEdge(const TaskEdge* edge) const {
  return out_edge2produced_regst.at(edge);
}

std::unique_ptr<TaskNode> TaskNode::CreateSameTypeNode() const {
  UNEXPECTED_RUN();
}

void TaskNode::InitWithFwNode(TaskNode* fw_node) {
  stage_node_ = fw_node->stage_node_;
  thrd_loc_id_ = fw_node->thrd_loc_id_;
  is_fw_node_ = false;
  related_fw_or_bp_node_ = fw_node;
}

void TaskNode::BindProducedRegstAndOutEdge(RegstDesc* regst,
                                           const TaskEdge* edge) {
  CHECK(produced_regst2out_edge.emplace(regst, edge).second);
  CHECK(out_edge2produced_regst.emplace(edge, regst).second);
}

void TaskNode::AddProducedRegstDesc(
    const std::string& regst_desc_name,
    std::unique_ptr<RegstDesc>&& regst_desc) {
  regst_desc->SetProducer(this);
  auto pair = std::make_pair(regst_desc_name, std::move(regst_desc));
  CHECK(produced_regst_descs_.insert(std::move(pair)).second);
}

void TaskNode::SubscribeRegstDescInnerPath() {
  for (const TaskEdge* edge : in_edges()) {
    RegstDesc* regst =  GetRelatedRegst(edge);
    Subscribe(regst);
  }
}

} // namespace oneflow
