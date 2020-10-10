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
#include "oneflow/core/job/thrd_id_generator.h"
#include "oneflow/core/graph/task_node.h"

namespace oneflow {

ThrdIdGenerator::ThrdIdGenerator(std::vector<std::pair<int64_t, TaskType>>& machine_task_type_vec,
                                 int64_t base_thrd_id)
    : base_thrd_id_(base_thrd_id) {
  HashMap<int64_t, std::set<TaskType>> machine2task_types;
  // machine_task_type = <machine_id, task_type>
  for (const auto& machine_task_type : machine_task_type_vec) {
    if (IsClassRegistered<int32_t, TickTockTaskType>(machine_task_type.second)) { continue; }
    if (TaskTypeThrdNumEqMax(machine_task_type.second,
                             machine_task_type2thrd_num_[machine_task_type])) {
      continue;
    }
    machine_task_type2thrd_num_[machine_task_type]++;
    machine2task_types[machine_task_type.first].emplace(machine_task_type.second);
  }

  InitLowerboundOfTaskType(machine2task_types);
}

int64_t ThrdIdGenerator::GenerateThrdId(int64_t machine_id, int64_t task_type) {
  if (IsClassRegistered<int32_t, TickTockTaskType>(task_type)) {
    return Global<IDMgr>::Get()->TickTockThrdId();
  }
  auto key = std::make_pair(machine_id, task_type);
  int64_t ret = machine_task_type2lowerbound_[key] + GetModThrdId(key);
  CHECK_GE(ret, 0);
  Global<IDMgr>::Get()->UpdateBaseIndependentThrdId(ret);
  return ret;
}

int64_t ThrdIdGenerator::GetModThrdId(std::pair<int64_t, int64_t> machine_task_type) {
  int64_t& offset = machine_task_type2offset_[machine_task_type];
  int64_t mod_thrd_id = offset % machine_task_type2thrd_num_[machine_task_type];
  offset++;
  return mod_thrd_id;
}

bool ThrdIdGenerator::TaskTypeThrdNumEqMax(int64_t task_type, int32_t thrd_num) {
  if (IsClassRegistered<int32_t, IndependentThreadNum4TaskType>(task_type)) {
    std::unique_ptr<IndependentThreadNum4TaskType> thread_num;
    thread_num.reset(NewObj<int32_t, IndependentThreadNum4TaskType>(task_type));
    return (thrd_num == *thread_num);
  } else {
    return false;
  }
}

void ThrdIdGenerator::InitLowerboundOfTaskType(
    const HashMap<int64_t, std::set<TaskType>>& machine2task_types) {
  for (const auto& pair : machine2task_types) {
    int64_t machine_id = pair.first;
    auto& task_types = pair.second;

    int64_t lowerbound = base_thrd_id_;
    for (int64_t task_type : task_types) {
      auto machine_task_type = std::make_pair(machine_id, task_type);
      machine_task_type2lowerbound_[machine_task_type] = lowerbound;
      lowerbound += machine_task_type2thrd_num_[machine_task_type];
    }
  }
}

}  // namespace oneflow
