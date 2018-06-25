#include "oneflow/core/graph/task_node.h"

namespace oneflow {

bool IsForwardTaskType(TaskType tt) {
  return tt == TaskType::kNormalForward || tt == TaskType::kRecurrentForward;
}

bool IsBackwardTaskType(TaskType tt) {
  return tt == TaskType::kNormalBackward || tt == TaskType::kRecurrentBackward;
}

bool IsMdUpdtTaskType(TaskType tt) { return tt == TaskType::kNormalMdUpdt; }

TaskNode::TaskNode()
    : machine_id_(-1),
      thrd_id_(-1),
      task_id_(-1),
      area_id_(0),
      chain_id_(-1),
      order_in_graph_(-1) {}

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

void TaskNode::set_area_id(int64_t val) {
  CHECK_EQ(area_id_, 0);
  area_id_ = val;
}

void TaskNode::set_chain_id(int64_t val) {
  CHECK_EQ(chain_id_, -1);
  chain_id_ = val;
}

void TaskNode::set_order_in_graph(int64_t val) {
  CHECK_EQ(order_in_graph_, -1);
  order_in_graph_ = val;
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
        return it->second->regst_desc_type().has_normal_regst_desc() && it->second->NumOfLbi() == 0;
      });
}

std::string TaskNode::VisualStr() const {
  std::stringstream ss;
  ss << TaskType_Name(GetTaskType()) << "\\n"
     << machine_id_ << ":" << thrd_id_ << "\\n"
     << task_id_;
  return ss.str();
}

bool TaskNode::IsMeaningLess() { return produced_regsts_.empty() && consumed_regsts_.empty(); }

void TaskNode::ToProto(TaskProto* task_proto) {
  task_proto->set_task_type(GetTaskType());
  task_proto->set_machine_id(machine_id_);
  task_proto->set_thrd_id(thrd_id_);
  task_proto->set_task_id(task_id_);
  task_proto->mutable_task_set_info()->set_area_id(area_id_);
  task_proto->mutable_task_set_info()->set_chain_id(chain_id_);
  task_proto->mutable_task_set_info()->set_order_in_graph(order_in_graph_);
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

int64_t TaskNode::MemZoneId121() const {
  const IDMgr* id_mgr = Global<IDMgr>::Get();
  if (device_type() == DeviceType::kCPU) {
    return id_mgr->CpuMemZoneId();
  } else {
    return id_mgr->GpuMemZoneId(id_mgr->GetGpuPhyIdFromThrdId(thrd_id_));
  }
}

void TaskNode::BuildCtrlRegstDescIfNeed(TaskNode* dst_node) {
  if (IsMeaningLess() || dst_node->IsMeaningLess()) return;
  const auto& dst_ancestors = dst_node->ancestors();
  if (dst_ancestors.find(this) != dst_ancestors.end()) return;
  RegstDescTypeProto regst_desc_type;
  regst_desc_type.mutable_ctrl_regst_desc();
  auto regst = NewProducedRegst(1, kMaxRegisterNum, regst_desc_type);
  std::string name = "out_ctrl_" + std::to_string(regst->regst_desc_id());
  CHECK(produced_regsts_.emplace(name, regst).second);
  dst_node->ConsumeRegst("in_ctrl", regst);
}

void TaskNode::BindEdgeWithProducedRegst(TaskEdge* edge, const std::string& name) {
  edge->AddRegst(name, GetProducedRegst(name));
}

std::shared_ptr<RegstDesc> TaskNode::ProduceRegst(const std::string& name) {
  return ProduceRegst(name, 1, kMaxRegisterNum);
}

std::shared_ptr<RegstDesc> TaskNode::ProduceRegst(const std::string& name, int32_t min_register_num,
                                                  int32_t max_register_num) {
  RegstDescTypeProto regst_desc_type;
  regst_desc_type.mutable_normal_regst_desc();
  return ProduceRegst(name, min_register_num, max_register_num, regst_desc_type);
}

std::shared_ptr<RegstDesc> TaskNode::NewProducedRegst(int32_t min_register_num,
                                                      int32_t max_register_num,
                                                      const RegstDescTypeProto& regst_desc_type) {
  auto regst = std::make_shared<RegstDesc>();
  regst->set_producer(this);
  *(regst->mut_regst_desc_type()) = regst_desc_type;
  regst->UpdtMinRegstNumIfNeed(min_register_num);
  regst->UpdtMaxRegstNumIfNeed(max_register_num);
  InitProducedRegstMemCase(regst.get());
  return regst;
}

std::shared_ptr<RegstDesc> TaskNode::ProduceRegst(const std::string& name, int32_t min_register_num,
                                                  int32_t max_register_num,
                                                  const RegstDescTypeProto& regst_desc_type) {
  auto regst = NewProducedRegst(min_register_num, max_register_num, regst_desc_type);
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
        Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(thrd_id_));
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
    RegstDesc* produced_regst = pair.second.get();
    produced_regst->UpdtMinRegstNumIfNeed(pair.second->MaxColNum());
    bool in_same_stream = true;
    for (const TaskNode* consumer : produced_regst->consumers()) {
      if (consumer->GlobalWorkStreamId() != GlobalWorkStreamId()) {
        in_same_stream = false;
        break;
      }
    }
    if (in_same_stream == false) {  // TODO: delete this hack
      if (produced_regst->max_register_num() >= 2) { produced_regst->UpdtMinRegstNumIfNeed(2); }
    }
  }
}

int64_t TaskNode::AllocateLocalWorkStreamId() {
  CHECK_NE(machine_id_, -1);
  CHECK_NE(thrd_id_, -1);
  if (UseIndependentWorkStream()) {
    if (device_type() == DeviceType::kCPU) {
      return 0;
    } else if (device_type() == DeviceType::kGPU) {
      return Global<IDMgr>::Get()->AllocateLocalWorkStreamId(machine_id_, thrd_id_);
    } else {
      UNIMPLEMENTED();
    }
  } else {
    return 0;
  }
}

void TaskNode::UpdateTaskId() {
  CHECK_NE(machine_id_, -1);
  CHECK_NE(thrd_id_, -1);
  task_id_ = Global<IDMgr>::Get()->NewTaskId(machine_id_, thrd_id_, AllocateLocalWorkStreamId());
}

int64_t TaskNode::LocalWorkStreamId() const {
  CHECK_NE(task_id_, -1);
  return Global<IDMgr>::Get()->LocalWorkStreamId4TaskId(task_id_);
}

int64_t TaskNode::GlobalWorkStreamId() const {
  CHECK_NE(task_id_, -1);
  return Global<IDMgr>::Get()->GlobalWorkStreamId4TaskId(task_id_);
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
    {kCopyCommNet, "9"},  {kBoxing, "10"},       {kPrint, "1"},          {kReduceScatter, "2"},
    {kReduceAdd, "2"},    {kReduceGather, "2"},
};

}  // namespace oneflow
