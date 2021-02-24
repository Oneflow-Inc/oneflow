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
#include "oneflow/core/graph/callback_notify_compute_task_node.h"

namespace oneflow {

REGISTER_INDEPENDENT_THREAD_NUM(TaskType::kCallbackNotify, 1);

REGISTER_COMPUTE_TASK_NODE_STREAM_INDEX_GETTER(DeviceType::kGPU, TaskType::kCallbackNotify)                       \
  .SetStreamIndexGetterFn([](const CompTaskNode* comp_task_node,                                                  \
                             std::function<uint32_t(int task_type)> Counter,                                      \
                             std::function<uint32_t(const TaskNode*)> AllocateCpuStreamIndexEvenly) -> uint32_t { \
      return CudaStreamIndex::kCompute;                                                                           \
  });

REGISTER_COMPUTE_TASK_NODE_STREAM_INDEX_GETTER(DeviceType::kCPU, TaskType::kCallbackNotify)                       \
  .SetStreamIndexGetterFn([](const CompTaskNode* comp_task_node,                                                  \
                             std::function<uint32_t(int task_type)> Counter,                                      \
                             std::function<uint32_t(const TaskNode*)> AllocateCpuStreamIndexEvenly) -> uint32_t { \
    if (comp_task_node->IsIndependent()) {                                                                        \
      return StreamIndex::Independent(TaskType::kCallbackNotify, Counter);                                        \
    } else {                                                                                                      \
      return AllocateCpuStreamIndexEvenly(comp_task_node);                                                        \
    }                                                                                                             \
  });

}  // namespace oneflow
