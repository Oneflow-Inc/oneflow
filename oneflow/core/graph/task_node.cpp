#include "oneflow/core/graph/task_node.h"

namespace oneflow {

TaskNode::TaskNode() : machine_id_(-1), thrd_loc_id_(-1), task_id_(-1) {}

void TaskNode::set_machine_id(int64_t val) {
  machine_id_ = val;
  if (thrd_loc_id_ != -1) { UpdateTaskId(); }
}

void TaskNode::set_thrd_loc_id(int64_t val) {
  thrd_loc_id_ = val;
  if (machine_id_ != -1) { UpdateTaskId(); }
}

std::shared_ptr<RegstDesc> TaskNode::GetProducedRegst(const std::string& name) {
  return produced_regsts_.at(name);
}

DeviceType TaskNode::device_type() const {
  return IDMgr::Singleton()->GetDeviceTypeFromThrdLocId(thrd_loc_id_);
}

void TaskNode::UpdateTaskId() {
  CHECK_NE(machine_id_, -1);
  CHECK_NE(thrd_loc_id_, -1);
  task_id_ = IDMgr::Singleton()->NewTaskId(machine_id_, thrd_loc_id_);
}

std::string TaskNode::VisualStr() const {
  std::stringstream ss;
  ss << TodoTaskType_Name(GetTaskType()) << "\\n"
     << machine_id_ << ":" << thrd_loc_id_ << "\\n"
     << task_id_;
  return ss.str();
}

std::shared_ptr<RegstDesc> TaskNode::ProduceRegst(const std::string& name,
                                                  int32_t min_register_num,
                                                  int32_t max_register_num) {
  auto regst = std::make_shared<RegstDesc>();
  regst->set_producer(this);
  regst->set_min_register_num(min_register_num);
  regst->set_max_register_num(max_register_num);
  CHECK(produced_regsts_.emplace(name, regst).second);
  return regst;
}

void TaskNode::ConsumeRegst(const std::string& name,
                            std::shared_ptr<RegstDesc> regst) {
  regst->AddConsumer(this);
  CHECK(consumed_regsts_.emplace(name, regst).second);
}

bool TaskNode::IsAllConsumedRegstLocked() {
  for (auto& pair : consumed_regsts_) {
    if (pair.second.lock()->IsLocked() == false) { return false; }
  }
  return true;
}

std::shared_ptr<RegstDesc> TaskEdge::GetRegst(
    const std::string& name_in_producer) const {
  return name_in_producer2regst_.at(name_in_producer).lock();
}

void TaskEdge::AddRegst(const std::string& name_in_producer,
                        std::shared_ptr<RegstDesc> regst) {
  CHECK(name_in_producer2regst_.emplace(name_in_producer, regst).second);
}

std::shared_ptr<RegstDesc> TaskEdge::GetSoleRegst() const {
  CHECK_EQ(name_in_producer2regst_.size(), 1);
  return name_in_producer2regst_.begin()->second.lock();
}

}  // namespace oneflow
