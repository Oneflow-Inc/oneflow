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
uint64_t& TaskNode::mut_thrd_loc_id() {
  CHECK(IsFwNode());
  return thrd_loc_id_;
}

void TaskNode::set_task_id() {
  uint64_t machine_id = stage_node_->machine_id();
  task_id_ = IDMgr::Singleton().NewTaskId(machine_id, thrd_loc_id_);
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

void TaskNode::BuildExecAndProducedRegsts(TaskGraph* gph) {
  if (IsFwNode()) {
    FwBuildExecAndProducedRegsts(gph);
  } else {
    BpBuildExecAndProducedRegsts(gph);
  }
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

void TaskNode::EnrollProducedRegstDesc(
    const std::string& regst_desc_name,
    std::unique_ptr<RegstDesc>&& regst_desc) {
  regst_desc->SetProducer(this);
  regst_desc->set_regst_desc_id(IDMgr::Singleton().NewRegstDescId(task_id_));
  auto pair = std::make_pair(regst_desc_name, std::move(regst_desc));
  CHECK(produced_regst_descs_.insert(std::move(pair)).second);
}

} // namespace oneflow
