#include "oneflow/core/graph/task_node.h"

namespace oneflow {

TaskNode::TaskNode() : produced_regst2out_edge_(11, [](const std::weak_ptr<RegstDesc>& v) { return std::hash<void*>() (v.lock().get()); }) {
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

std::shared_ptr<RegstDesc> TaskNode::GetProducedRegstDesc(
    const std::string& regst_desc_name) {
  auto it = produced_regst_descs_.find(regst_desc_name);
  if (it == produced_regst_descs_.end()) {
    return nullptr;
  } else {
    return it->second;
  }
}

std::shared_ptr<RegstDesc> TaskNode::GetSubscribedRegstDesc(
    const std::string& regst_desc_name) const {
  auto it = subscribed_regst_descs_.find(regst_desc_name);
  if (it == subscribed_regst_descs_.end()) {
    return nullptr;
  } else {
    return it->second.lock();
  }
}

void TaskNode::TakeOverRegstDesc(TaskNode* rhs,
                                 const std::string& regst_desc_name) {
  CHECK_EQ(stage_node_->machine_id(), rhs->stage_node_->machine_id());
  CHECK_EQ(thrd_loc_id_, rhs->thrd_loc_id_);
  std::shared_ptr<RegstDesc> this_regst;
  auto rhs_regst_it = rhs->produced_regst_descs_.find(regst_desc_name);
  CHECK_EQ(produced_regst2out_edge_.count(rhs_regst_it->second), 0);
  this_regst.swap(rhs_regst_it->second);
  this_regst->SetProducer(this);
  this_regst->set_regst_desc_id(IDMgr::Singleton().NewRegstDescId(task_id_));
  rhs->produced_regst_descs_.erase(rhs_regst_it);
  CHECK(produced_regst_descs_.emplace(regst_desc_name, this_regst).second);
}

void TaskNode::EraseProducedEmptyRegsts() {
  EraseIf<std::string, std::shared_ptr<RegstDesc>> (&produced_regst_descs_, []
      (HashMap<std::string, std::shared_ptr<RegstDesc>>::iterator it) {
    return it->second->lbn2shape().empty();
  });
}

void TaskNode::EraseZeroSizeBlobInProducedRegsts() {
  for (const auto& pair : produced_regst_descs_) {
    pair.second->EraseZeroSizeBlob();
  }
}

const TaskEdge* TaskNode::GetOutEdge4ProducedRegst(
    std::weak_ptr<RegstDesc> regst) const {
  return produced_regst2out_edge_.at(regst);
}

std::shared_ptr<RegstDesc> TaskNode::GetProducedRegst4OutEdge(
    const TaskEdge* edge) const {
  return out_edge2produced_regst_.at(edge).lock();
}

void TaskNode::InitWithFwNode(TaskNode* fw_node) {
  stage_node_ = fw_node->stage_node_;
  thrd_loc_id_ = fw_node->thrd_loc_id_;
  is_fw_node_ = false;
  related_fw_or_bp_node_ = fw_node;
  set_task_id();
}

void TaskNode::BindProducedRegstAndOutEdge(std::weak_ptr<RegstDesc> regst,
                                           const TaskEdge* edge) {
  CHECK(produced_regst2out_edge_.emplace(regst, edge).second);
  CHECK(out_edge2produced_regst_.emplace(edge, regst).second);
}

std::shared_ptr<RegstDesc> TaskNode::NewProducedRegstDesc(
    const std::string& regst_desc_name) {
  auto regst_desc = std::make_shared<RegstDesc> ();
  regst_desc->SetProducer(this);
  regst_desc->set_regst_desc_id(IDMgr::Singleton().NewRegstDescId(task_id_));
  CHECK(produced_regst_descs_.emplace(regst_desc_name, regst_desc).second);
  return regst_desc;
}

void TaskNode::SubscribeRegstDesc(const std::string& regst_desc_name,
                                  std::shared_ptr<RegstDesc> regst_desc) {
  CHECK(subscribed_regst_descs_.emplace(regst_desc_name, regst_desc).second);
  regst_desc->AddSubscriber(this);
}

void TaskNode::ToProto(TaskProto* ret) const {
  ret->set_id(task_id_);
  ret->set_type(task_type());
  ret->set_machine_id(stage_node_->machine_id());
  ret->set_thrd_local_id(thrd_loc_id_);
  ret->set_is_forward(is_fw_node_);
  exec_gph_.ToExecSequence(ret->mutable_exec_sequence());
  for (const auto& pair : produced_regst_descs_) {
    RegstDescProto regst_desc_proto;
    pair.second->ToProto(&regst_desc_proto);
    CHECK(ret->mutable_produced_regst_desc()->insert(
          {pair.first, regst_desc_proto}).second);
  }
  for (const auto& pair : subscribed_regst_descs_) {
    auto regst_desc = pair.second.lock();
    if (regst_desc) {
      CHECK(ret->mutable_subscribed_regst_desc_id()->insert(
          {pair.first, regst_desc->regst_desc_id()}).second);
    }
  }
}

std::string TaskNode::VisualStr() const {
  std::stringstream ss;
  ss << (is_fw_node_ ? "Fw" : "Bp");
  ss << node_id_str() << "_";
  return ss.str();
}

std::string TaskNode::DebugStr() const {
  std::stringstream ss;
  ss << "{" << node_id_str() << "\t";
  for (const auto& pair : produced_regst_descs_) {
    ss << "{" << pair.first << ":" << pair.second->DebugStr() << "}"; 
  }
  ss << "}";
  return ss.str();
}

} // namespace oneflow
