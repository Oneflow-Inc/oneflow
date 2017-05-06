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

RegstDesc* TaskNode::GetProducedRegstDesc(const std::string& regst_desc_name) {
  return produced_regst_descs_.at(regst_desc_name).get();
}

void TaskNode::TakeOverRegstDesc(TaskNode* rhs,
                                 const std::string& regst_desc_name) {
  std::unique_ptr<RegstDesc> this_regst;
  auto rhs_regst_it = rhs->produced_regst_descs_.find(regst_desc_name);
  this_regst.swap(rhs_regst_it->second);
  this_regst->SetProducer(this);
  this_regst->set_regst_desc_id(IDMgr::Singleton().NewRegstDescId(task_id_));
  rhs->produced_regst_descs_.erase(rhs_regst_it);
  CHECK(rhs->forwarded_regst_descs_.emplace(regst_desc_name,
                                            this_regst.get()).second);
  CHECK(produced_regst_descs_.emplace(regst_desc_name,
                                      std::move(this_regst)).second);
}

const RegstDesc* TaskNode::ForwardedRegstDesc(
    const std::string& regst_desc_name) const {
  return forwarded_regst_descs_.at(regst_desc_name);
}

const TaskEdge* TaskNode::GetOutEdge4ProducedRegst(RegstDesc* regst) const {
  return produced_regst2out_edge.at(regst);
}

RegstDesc* TaskNode::GetProducedRegst4OutEdge(const TaskEdge* edge) const {
  return out_edge2produced_regst.at(edge);
}

void TaskNode::InitWithFwNode(TaskNode* fw_node) {
  stage_node_ = fw_node->stage_node_;
  thrd_loc_id_ = fw_node->thrd_loc_id_;
  is_fw_node_ = false;
  related_fw_or_bp_node_ = fw_node;
  set_task_id();
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
  CHECK(produced_regst_descs_.emplace(regst_desc_name, std::move(regst_desc)).second);
}

TaskProto TaskNode::ToProto() const {
  TaskProto task_proto;
  task_proto.set_id(task_id_);
  task_proto.set_machine_id(stage_node_->machine_id());
  task_proto.set_thrd_local_id(thrd_loc_id_);
  task_proto.set_is_forward(is_fw_node_);
  *task_proto.mutable_exec_graph() = exec_gph_.ToProto();
  for (const auto& pair : produced_regst_descs_) {
    task_proto.mutable_produced_regst_desc_ids()->Add(
        pair.second->regst_desc_id());
  }
  for (const auto& pair : subscribed_regst_descs_) {
    task_proto.mutable_subscribed_regst_desc_ids()->Add(
        pair.second->regst_desc_id());
  }
  return task_proto;
}

} // namespace oneflow
