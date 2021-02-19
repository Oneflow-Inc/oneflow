/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/graph/task_node.h"

namespace oneflow {

namespace {

void ForEachDataEdge(const std::unordered_set<TaskEdge*>& edges,
                     const std::function<void(TaskEdge*)>& Handler) {
  for (TaskEdge* edge : edges) {
    const auto& regsts = edge->GetRegsts();
    int32_t data_regst_size =
        std::count_if(regsts.begin(), regsts.end(), [](const std::shared_ptr<RegstDesc>& regst) {
          return regst->regst_desc_type().has_data_regst_desc();
        });
    if (data_regst_size == regsts.size()) {
      Handler(edge);
    } else {
      CHECK_EQ(data_regst_size, 0);
    }
  }
}

}  // namespace

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

const std::list<std::shared_ptr<RegstDesc>>& TaskNode::GetConsumedRegst(const std::string& name) {
  return consumed_regsts_.at(name);
}

std::shared_ptr<RegstDesc> TaskNode::GetSoleConsumedRegst(const std::string& name) {
  auto it = consumed_regsts_.find(name);
  if (it == consumed_regsts_.end()) { return nullptr; }
  const std::list<std::shared_ptr<RegstDesc>>& vec = it->second;
  CHECK_EQ(vec.size(), 1);
  return vec.front();
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
  CHECK_GE(thrd_id_, 0);
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
    for (const std::shared_ptr<RegstDesc>& regst : pair.second) {
      PinConsumedRegstMemCase(regst->mut_mem_case());
    }
  }
}

void TaskNode::NaiveInferProducedDataRegstTimeShape() {
  if (IsMeaningLess()) { return; }
  std::shared_ptr<Shape> time_shape;
  ForEachConsumedDataRegst([&time_shape](const std::string& name, const RegstDesc* regst) {
    if (time_shape) {
      CHECK_EQ(*time_shape.get(), *regst->data_regst_time_shape().get());
    } else {
      time_shape = regst->data_regst_time_shape();
    }
  });

  CHECK(time_shape);

  ForEachProducedDataRegst([time_shape](const std::string& name, RegstDesc* regst) {
    *regst->mut_data_regst_time_shape() = time_shape;
  });
}

void TaskNode::InferTimeShapeIfMeaningful() {
  if (!IsMeaningLess()) { InferProducedDataRegstTimeShape(); }
}

std::shared_ptr<Shape> TaskNode::GetFastestInputOutputTimeShape() const {
  std::shared_ptr<Shape> shape;
  auto UpdateRetShape = [&](TaskEdge* edge) {
    for (const auto& regst : edge->GetRegsts()) {
      if (!shape || shape->elem_cnt() < regst->data_regst_time_shape()->elem_cnt()) {
        shape = regst->data_regst_time_shape();
      }
    }
  };
  ForEachOutDataEdge(UpdateRetShape);
  if (shape) { return shape; }
  ForEachInDataEdge(UpdateRetShape);
  return shape;
}

void TaskNode::ForEachConsumedDataRegst(
    const std::function<void(const std::string&, const RegstDesc*)>& Handler) const {
  for (const auto& pair : consumed_regsts_) {
    for (const auto& regst : pair.second) {
      if (!regst->regst_desc_type().has_data_regst_desc()) { continue; }
      Handler(pair.first, regst.get());
    }
  }
}

void TaskNode::ForEachProducedDataRegst(
    const std::function<void(const std::string&, RegstDesc*)>& Handler) {
  for (auto& pair : produced_regsts_) {
    if (!pair.second->regst_desc_type().has_data_regst_desc()) { continue; }
    Handler(pair.first, pair.second.get());
  }
}

void TaskNode::Build() {
  if (consumed_regsts_.size()) { CHECK(IsReadyForBuild()); }
  BuildExecGphAndRegst();
  LockRegsts();
  FixRegisterNumRange();
}

void TaskNode::EraseZeroSizeProducedBlob() {
  for (auto& pair : produced_regsts_) { pair.second->EraseZeroSizeBlob(); }
}

void TaskNode::EraseZeroSizeConsumedRegst() {
  for (auto& pair : consumed_regsts_) {
    for (auto it = pair.second.begin(); it != pair.second.end();) {
      auto regst_ptr = *it;
      CHECK(regst_ptr);
      if (regst_ptr->regst_desc_type().has_data_regst_desc() && regst_ptr->NumOfLbi() == 0) {
        it = pair.second.erase(it);
      } else {
        ++it;
      }
    }
  }
  EraseIf<std::string, std::list<std::shared_ptr<RegstDesc>>>(
      &consumed_regsts_,
      [](HashMap<std::string, std::list<std::shared_ptr<RegstDesc>>>::iterator it) {
        return it->second.empty();
      });
}

void TaskNode::EraseZeroSizeProducedRegst() {
  EraseIf<std::string, std::shared_ptr<RegstDesc>>(
      &produced_regsts_, [](HashMap<std::string, std::shared_ptr<RegstDesc>>::iterator it) {
        return it->second->regst_desc_type().has_data_regst_desc() && it->second->NumOfLbi() == 0;
      });
}

void TaskNode::UnbindBnWithEmptyRegst() {
  exec_gph_.ForEachNode([&](ExecNode* exec_node) { exec_node->UnbindBnWithEmptyRegst(); });
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
  CHECK_NE(chain_id_, -1);
  task_proto->set_task_type(GetTaskType());
  task_proto->set_machine_id(machine_id_);
  task_proto->set_thrd_id(thrd_id_);
  task_proto->set_task_id(task_id_);
  task_proto->set_job_id(GlobalJobDesc().job_id());
  task_proto->mutable_task_set_info()->set_area_id(area_id_);
  task_proto->mutable_task_set_info()->set_chain_id(chain_id_);
  task_proto->mutable_task_set_info()->set_order_in_graph(order_in_graph_);
  exec_gph_.ToExecSequence(parallel_ctx(), task_proto->mutable_exec_sequence());
  auto produced_regst_proto = task_proto->mutable_produced_regst_desc();
  for (auto& pair : produced_regsts_) {
    RegstDescProto regst_desc_proto;
    pair.second->ToProto(&regst_desc_proto);
    CHECK(produced_regst_proto->insert({pair.first, regst_desc_proto}).second);
  }
  auto consumed_regst_proto = task_proto->mutable_consumed_regst_desc_id();
  for (const auto& pair : consumed_regsts_) {
    RegstDescIdSet regst_desc_ids;
    for (const std::shared_ptr<RegstDesc>& regst : pair.second) {
      regst_desc_ids.add_regst_desc_id(regst->regst_desc_id());
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
  if (IsMeaningLess() || dst_node->IsMeaningLess()) { return; }
  for (const TaskEdge* in_edge : dst_node->in_edges()) {
    if (in_edge->src_node() == this) { return; }
  }
  BuildCtrlRegstDesc(dst_node);
}

RegstDesc* TaskNode::BuildCtrlRegstDesc(TaskNode* dst_node) {
  std::string name;
  return BuildCtrlRegstDesc(dst_node, &name);
}

RegstDesc* TaskNode::BuildCtrlRegstDesc(TaskNode* dst_node, std::string* name) {
  RegstDescTypeProto regst_desc_type;
  regst_desc_type.mutable_ctrl_regst_desc();
  auto regst = NewProducedRegst(false, 1, kMaxRegisterNum, regst_desc_type);
  *name = "out_ctrl_" + std::to_string(regst->regst_desc_id());
  CHECK(produced_regsts_.emplace(*name, regst).second);
  dst_node->ConsumeRegst("in_ctrl", regst);
  return regst.get();
}

void TaskNode::BindEdgeWithProducedRegst(TaskEdge* edge, const std::string& name) {
  edge->AddRegst(name, GetProducedRegst(name));
}

std::shared_ptr<RegstDesc> TaskNode::ProduceRegst(const std::string& name, bool enable_reuse_mem) {
  return ProduceRegst(name, enable_reuse_mem, 1, kMaxRegisterNum);
}

std::shared_ptr<RegstDesc> TaskNode::ProduceRegst(const std::string& name, bool enable_reuse_mem,
                                                  int32_t min_register_num,
                                                  int32_t max_register_num) {
  RegstDescTypeProto regst_desc_type;
  regst_desc_type.mutable_data_regst_desc();
  return ProduceRegst(name, enable_reuse_mem, min_register_num, max_register_num, regst_desc_type);
}

std::shared_ptr<RegstDesc> TaskNode::ProduceRegst(const std::string& name, bool enable_reuse_mem,
                                                  int32_t min_register_num,
                                                  int32_t max_register_num,
                                                  const RegstDescTypeProto& regst_desc_type) {
  auto regst =
      NewProducedRegst(enable_reuse_mem, min_register_num, max_register_num, regst_desc_type);
  CHECK(produced_regsts_.emplace(name, regst).second);
  return regst;
}

std::shared_ptr<RegstDesc> TaskNode::NewProducedRegst(bool enable_reuse_mem,
                                                      int32_t min_register_num,
                                                      int32_t max_register_num,
                                                      const RegstDescTypeProto& regst_desc_type) {
  auto regst = std::make_shared<RegstDesc>();
  regst->set_producer(this);
  *(regst->mut_regst_desc_type()) = regst_desc_type;
  regst->UpdtMinRegstNumIfNeed(min_register_num);
  regst->UpdtMaxRegstNumIfNeed(max_register_num);
  regst->set_enable_reuse_mem(GlobalJobDesc().enable_reuse_mem() && enable_reuse_mem);
  InitProducedRegstMemCase(regst.get());
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
    mem_case->mutable_host_mem()->mutable_cuda_pinned_mem()->set_device_id(GpuPhyId());
  }
}

void TaskNode::ConsumeRegst(const std::string& name) {
  consumed_regsts_.emplace(name, std::list<std::shared_ptr<RegstDesc>>{});
}

void TaskNode::ConsumeRegst(const std::string& name, const std::shared_ptr<RegstDesc>& regst) {
  regst->AddConsumer(this);
  consumed_regsts_[name].push_back(regst);
}

bool TaskNode::IsAllConsumedDataRegstLocked() {
  for (const auto& pair : consumed_regsts_) {
    for (const std::shared_ptr<RegstDesc>& regst_desc : pair.second) {
      if (regst_desc->regst_desc_type().has_data_regst_desc() && regst_desc->IsLocked() == false) {
        return false;
      }
    }
  }
  return true;
}

void TaskNode::TryLockConsumedRegst(const std::string& name) {
  auto consumed_regsts_it = consumed_regsts_.find(name);
  if (consumed_regsts_it == consumed_regsts_.end()) { return; }
  for (const std::shared_ptr<RegstDesc>& wrd : consumed_regsts_it->second) {
    const std::shared_ptr<RegstDesc>& srd = wrd;
    if (srd->IsLocked() == false) { srd->Lock(); }
  }
}

void TaskNode::LockRegsts() {
  for (auto& pair : produced_regsts_) { pair.second->Lock(); }
}

void TaskNode::FixRegisterNumRange() {
  for (auto& pair : produced_regsts_) {
    RegstDesc* produced_regst = pair.second.get();
    bool in_same_stream = true;
    for (const TaskNode* consumer : produced_regst->consumers()) {
      if (consumer->GlobalWorkStreamId() != GlobalWorkStreamId()) {
        in_same_stream = false;
        break;
      }
    }
    if (in_same_stream == false && area_id_ != static_cast<int64_t>(kMdUpdtArea)
        && GetTaskType() == TaskType::kCopyHd) {  // TODO: delete this hack
      if (produced_regst->max_register_num() >= 2) { produced_regst->UpdtMinRegstNumIfNeed(2); }
    }
  }
}

void TaskNode::UpdateTaskId() {
  CHECK_NE(machine_id_, -1);
  CHECK_NE(thrd_id_, -1);
  task_id_ = Global<IDMgr>::Get()->NewTaskId(machine_id_, thrd_id_);
}

int64_t TaskNode::GlobalWorkStreamId() const {
  CHECK_NE(task_id_, -1);
  return Global<IDMgr>::Get()->GlobalWorkStreamId4TaskId(task_id_);
}

void TaskNode::EraseConsumedRegstsByName(const std::string& name) {
  if (consumed_regsts_.find(name) != consumed_regsts_.end()) {
    for (auto& regst : consumed_regsts_[name]) { regst->DeleteConsumer(this); }
    CHECK_EQ(consumed_regsts_.erase(name), 1);
  }
}

std::shared_ptr<RegstDesc> TaskEdge::GetRegst(const std::string& name_in_producer) const {
  return name_in_producer2regst_.at(name_in_producer);
}

std::shared_ptr<RegstDesc> TaskEdge::GetSoleRegst() const {
  CHECK_EQ(name_in_producer2regst_.size(), 1);
  return name_in_producer2regst_.begin()->second;
}

std::vector<std::shared_ptr<RegstDesc>> TaskEdge::GetRegsts() const {
  std::vector<std::shared_ptr<RegstDesc>> regst_descs;
  for (auto& pair : name_in_producer2regst_) { regst_descs.emplace_back(pair.second); }
  return regst_descs;
}

void TaskEdge::AddRegst(const std::string& name_in_producer,
                        const std::shared_ptr<RegstDesc>& regst) {
  CHECK(name_in_producer2regst_.emplace(name_in_producer, regst).second);
}

RegstDescProto* FindOrCreateProducedCtrlRegstDesc(TaskProto* task_proto,
                                                  const std::string& regst_desc_name) {
  auto* produced_regst_desc = task_proto->mutable_produced_regst_desc();
  if (produced_regst_desc->find(regst_desc_name) == produced_regst_desc->end()) {
    RegstDescProto ctrl_regst_desc;
    InitCtrlRegstDesc(task_proto->task_id(), &ctrl_regst_desc);
    CHECK(produced_regst_desc->insert({regst_desc_name, ctrl_regst_desc}).second);
  }
  return &produced_regst_desc->at(regst_desc_name);
}

RegstDescIdSet* FindOrCreateConsumedCtrlRegstDescIdSet(TaskProto* task_proto,
                                                       const std::string& regst_desc_name) {
  auto* consumed_regst_desc_id_sets = task_proto->mutable_consumed_regst_desc_id();
  if (consumed_regst_desc_id_sets->find(regst_desc_name) == consumed_regst_desc_id_sets->end()) {
    CHECK(consumed_regst_desc_id_sets->insert({regst_desc_name, RegstDescIdSet()}).second);
  }
  return &consumed_regst_desc_id_sets->at(regst_desc_name);
}

void TaskNode::ForEachInDataEdge(const std::function<void(TaskEdge*)>& Handler) const {
  ForEachDataEdge(in_edges(), Handler);
}

void TaskNode::ForEachOutDataEdge(const std::function<void(TaskEdge*)>& Handler) const {
  ForEachDataEdge(out_edges(), Handler);
}

void TaskNode::ForEachNodeOnInDataEdge(const std::function<void(TaskNode*)>& Handler) const {
  ForEachInDataEdge([&](TaskEdge* in_edge) { Handler(in_edge->src_node()); });
}

void TaskNode::ForEachNodeOnOutDataEdge(const std::function<void(TaskNode*)>& Handler) const {
  ForEachOutDataEdge([&](TaskEdge* out_edge) { Handler(out_edge->dst_node()); });
}

void TaskNode::ForEachNodeOnInOutDataEdge(const std::function<void(TaskNode*)>& Handler) const {
  ForEachNodeOnInDataEdge(Handler);
  ForEachNodeOnOutDataEdge(Handler);
}

TaskEdge* TaskNode::GetSoleEdge(void (TaskNode::*ForEachEdge)(const std::function<void(TaskEdge*)>&)
                                    const) const {
  TaskEdge* ret = nullptr;
  (this->*ForEachEdge)([&](TaskEdge* edge) {
    CHECK(ret == nullptr);
    ret = edge;
  });
  CHECK_NOTNULL(ret);
  return ret;
}

size_t TaskNode::GetEdgesSize(void (TaskNode::*ForEachEdge)(const std::function<void(TaskEdge*)>&)
                                  const) const {
  size_t size = 0;
  (this->*ForEachEdge)([&](TaskEdge* edge) { ++size; });
  return size;
}

TaskEdge* TaskNode::SoleInDataEdge() const { return GetSoleEdge(&TaskNode::ForEachInDataEdge); }

TaskEdge* TaskNode::SoleOutDataEdge() const { return GetSoleEdge(&TaskNode::ForEachOutDataEdge); }

size_t TaskNode::in_data_edges_size() const { return GetEdgesSize(&TaskNode::ForEachInDataEdge); }

size_t TaskNode::out_data_edges_size() const { return GetEdgesSize(&TaskNode::ForEachOutDataEdge); }

}  // namespace oneflow
