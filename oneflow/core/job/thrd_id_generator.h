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
#ifndef ONEFLOW_CORE_JOB_THRD_ID_GENERATOR_H_
#define ONEFLOW_CORE_JOB_THRD_ID_GENERATOR_H_

#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class ThrdIdGenerator final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ThrdIdGenerator);

  ThrdIdGenerator(std::vector<std::pair<int64_t, TaskType>>& machine_task_type_vec,
                  int64_t base_thrd_id);

  int64_t GenerateThrdId(int64_t machine_id, int64_t task_type);

 private:
  int64_t GetModThrdId(std::pair<int64_t, int64_t> machine_task_type);
  bool TaskTypeThrdNumEqMax(int64_t task_type, int32_t thrd_num);
  void InitLowerboundOfTaskType(const HashMap<int64_t, std::set<TaskType>>& machine2task_types);

  int64_t tick_tock_thrd_id_;
  int64_t base_thrd_id_;
  HashMap<std::pair<int64_t, int64_t>, int32_t> machine_task_type2thrd_num_;
  HashMap<std::pair<int64_t, int64_t>, int64_t> machine_task_type2offset_;
  HashMap<std::pair<int64_t, int64_t>, int32_t> machine_task_type2lowerbound_;
};
}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_THRD_ID_GENERATOR_H_
