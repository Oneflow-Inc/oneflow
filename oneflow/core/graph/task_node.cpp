#include "oneflow/core/graph/task_node.h"

namespace oneflow {

bool IsForwardTaskType(TaskType tt) {
  return tt == TaskType::kNormalForward || tt == TaskType::kRecurrentForward;
}

bool IsBackwardTaskType(TaskType tt) {
  return tt == TaskType::kNormalBackward || tt == TaskType::kRecurrentBackward;
}

bool IsMdUpdtTaskType(TaskType tt) { return tt == TaskType::kNormalMdUpdt; }

TaskNode::TaskNode() : machine_id_(-1), thrd_id_(-1), task_id_(-1) {}

std::shared_ptr<RegstDesc> TaskNode::GetProducedRegst(const std::string& name) {
  auto produced_regsts_it = produced_regsts_.find(name);
  if (produced_regsts_it == produced_regsts_.end()) {
    return nullptr;
  } else {
    return produced_regsts_it->second;
  }
}

const std::list<std::weak_ptr<RegstDesc>>& TaskNode::GetConsumedRegst(const std::string& name) {
  return consumed_regsts_.at(name);
}

std::shared_ptr<RegstDesc> TaskNode::GetSoleConsumedRegst(const std::string& name) {
  auto it = consumed_regsts_.find(name);
  if (it == consumed_regsts_.end()) { return nullptr; }
  const std::list<std::weak_ptr<RegstDesc>>& vec = it->second;
  CHECK_EQ(vec.size(), 1);
  return vec.front().lock();
}

DeviceType TaskNode::device_type() const {
  return Global<IDMgr>::Get()->GetDeviceTypeFromThrdId(thrd_id_);
}

void TaskNode::set_machine_id(int64_t val) {
  CHECK_EQ(machine_id_, -1);
  machine_id_ = val;
  if (thrd_id_ != -1) { UpdateTaskId(); }
}

void TaskNode::set_thrd_id(int64_t val) {
  CHECK_EQ(thrd_id_, -1);
  thrd_id_ = val;
  if (machine_id_ != -1) { UpdateTaskId(); }
}

void TaskNode::PinConsumedRegst() {
  for (auto& pair : consumed_regsts_) {
    for (std::weak_ptr<RegstDesc> regst : pair.second) {
      PinConsumedRegstMemCase(regst.lock()->mut_mem_case());
    }
  }
}

void TaskNode::Build() {
  BuildExecGphAndRegst();
  LockRegsts();
  FixRegisterNumRange();
}

void TaskNode::EraseEmptyProducedRegst() {
  for (auto& pair : produced_regsts_) { pair.second->EraseZeroSizeBlob(); }
  EraseIf<std::string, std::shared_ptr<RegstDesc>>(
      &produced_regsts_, [](HashMap<std::string, std::shared_ptr<RegstDesc>>::iterator it) {
        return it->second->NumOfLbi() == 0;
      });
}

std::string TaskNode::VisualStr() const {
  std::stringstream ss;
  ss << TaskType_Name(GetTaskType()) << "\\n"
     << machine_id_ << ":" << thrd_id_ << "\\n"
     << task_id_;
  return ss.str();
}

bool TaskNode::IsMeaningLess() {
  ClearOutOfDateConsumedRegst();
  return produced_regsts_.empty() && consumed_regsts_.empty();
}

void TaskNode::ToProto(TaskProto* task_proto) {
  task_proto->set_task_type(GetTaskType());
  task_proto->set_machine_id(machine_id_);
  task_proto->set_thrd_id(thrd_id_);
  task_proto->set_task_id(task_id_);
  exec_gph_.ToExecSequence(IsBackwardTaskType(GetTaskType()) == false, parallel_ctx(),
                           task_proto->mutable_exec_sequence());
  auto produced_regst_proto = task_proto->mutable_produced_regst_desc();
  for (auto& pair : produced_regsts_) {
    RegstDescProto regst_desc_proto;
    pair.second->ToProto(&regst_desc_proto);
    CHECK(produced_regst_proto->insert({pair.first, regst_desc_proto}).second);
  }
  ClearOutOfDateConsumedRegst();
  auto consumed_regst_proto = task_proto->mutable_consumed_regst_desc_id();
  for (const auto& pair : consumed_regsts_) {
    RegstDescIdSet regst_desc_ids;
    for (std::weak_ptr<RegstDesc> regst : pair.second) {
      regst_desc_ids.add_regst_desc_id(regst.lock()->regst_desc_id());
    }
    CHECK(consumed_regst_proto->insert({pair.first, regst_desc_ids}).second);
  }
}

void TaskNode::BindEdgeWithProducedRegst(TaskEdge* edge, const std::string& name) {
  edge->AddRegst(name, GetProducedRegst(name));
}

std::shared_ptr<RegstDesc> TaskNode::ProduceRegst(const std::string& name) {
  return ProduceRegst(name, 1, kMaxRegisterNum);
}

std::shared_ptr<RegstDesc> TaskNode::ProduceRegst(const std::string& name, int32_t min_register_num,
                                                  int32_t max_register_num) {
  auto regst = std::make_shared<RegstDesc>();
  regst->set_producer(this);
  regst->UpdtMinRegstNumIfNeed(min_register_num);
  regst->UpdtMaxRegstNumIfNeed(max_register_num);
  InitProducedRegstMemCase(regst.get());
  CHECK(produced_regsts_.emplace(name, regst).second);
  return regst;
}

void TaskNode::InitProducedRegstMemCase(RegstDesc* regst) {
  InitProducedRegstMemCase(regst->mut_mem_case());
}

void TaskNode::InitProducedRegstMemCase(MemoryCase* mem_case) {
  if (device_type() == DeviceType::kCPU) {
    mem_case->mutable_host_mem();
  } else if (device_type() == DeviceType::kGPU) {
    mem_case->mutable_device_cuda_mem()->set_device_id(
        Global<IDMgr>::Get()->GetGpuDevPhyIdFromThrdId(thrd_id_));
  } else {
    UNIMPLEMENTED();
  }
}

void TaskNode::PinConsumedRegstMemCase(MemoryCase* mem_case) {
  if (mem_case->has_host_mem() && device_type() == DeviceType::kGPU) {
    mem_case->mutable_host_mem()->set_used_by_device(true);
  }
}

void TaskNode::ConsumeRegst(const std::string& name, std::shared_ptr<RegstDesc> regst) {
  regst->AddConsumer(this);
  consumed_regsts_[name].push_back(regst);
}

bool TaskNode::IsAllConsumedRegstLocked() {
  for (const auto& pair : consumed_regsts_) {
    for (std::weak_ptr<RegstDesc> regst_desc : pair.second) {
      if (regst_desc.lock()->IsLocked() == false) { return false; }
    }
  }
  return true;
}

void TaskNode::TryLockConsumedRegst(const std::string& name) {
  auto consumed_regsts_it = consumed_regsts_.find(name);
  if (consumed_regsts_it == consumed_regsts_.end()) { return; }
  for (std::weak_ptr<RegstDesc> wrd : consumed_regsts_it->second) {
    std::shared_ptr<RegstDesc> srd = wrd.lock();
    if (srd->IsLocked() == false) { srd->Lock(); }
  }
}

void TaskNode::LockRegsts() {
  for (auto& pair : produced_regsts_) { pair.second->Lock(); }
}

void TaskNode::FixRegisterNumRange() {
  for (auto& pair : produced_regsts_) {
    pair.second->UpdtMinRegstNumIfNeed(pair.second->MaxColNum());
  }
}

void TaskNode::UpdateTaskId() {
  CHECK_NE(machine_id_, -1);
  CHECK_NE(thrd_id_, -1);
  task_id_ = Global<IDMgr>::Get()->NewTaskId(machine_id_, thrd_id_);
}

void TaskNode::ClearOutOfDateConsumedRegst() {
  for (auto& pair : consumed_regsts_) {
    for (auto it = pair.second.begin(); it != pair.second.end();) {
      if (it->lock() == nullptr) {
        pair.second.erase(it++);
      } else {
        ++it;
      }
    }
  }
  EraseIf<std::string, std::list<std::weak_ptr<RegstDesc>>>(
      &consumed_regsts_,
      [](HashMap<std::string, std::list<std::weak_ptr<RegstDesc>>>::iterator it) {
        return it->second.empty();
      });
}

std::shared_ptr<RegstDesc> TaskEdge::GetRegst(const std::string& name_in_producer) const {
  return name_in_producer2regst_.at(name_in_producer).lock();
}

std::shared_ptr<RegstDesc> TaskEdge::GetSoleRegst() const {
  CHECK_EQ(name_in_producer2regst_.size(), 1);
  return name_in_producer2regst_.begin()->second.lock();
}

void TaskEdge::AddRegst(const std::string& name_in_producer, std::shared_ptr<RegstDesc> regst) {
  CHECK(name_in_producer2regst_.emplace(name_in_producer, regst).second);
}

std::map<TaskType, std::string> task_type2color = {
    {kInvalid, "0"},      {kNormalForward, "2"}, {kNormalBackward, "3"}, {kRecordLoad, "1"},
    {kDecode, "1"},       {kLoss, "4"},          {kLossAcc, "5"},        {kLossPrint, "1"},
    {kNormalMdUpdt, "6"}, {kMdSave, "1"},        {kMdDiffAcc, "7"},      {kCopyHd, "8"},
    {kCopyCommNet, "9"},  {kBoxing, "10"},       {kPrint, "1"},
};

}  // namespace oneflow
