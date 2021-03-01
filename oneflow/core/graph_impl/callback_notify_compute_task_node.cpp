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

REGISTER_COMPUTE_TASK_NODE_STREAM_INDEX_GETTER(DeviceType::kGPU, TaskType::kCallbackNotify)
    .SetStreamIndexGetterFn([](DeviceId device_id) -> uint32_t {
      auto* cuda_stream_index_generator = dynamic_cast<CudaStreamIndexGenerator*>(
          Global<IDMgr>::Get()->GetStreamIndexGeneratorManager()->GetGenerator(device_id));
      CHECK_NOTNULL(cuda_stream_index_generator);
      return cuda_stream_index_generator->GenerateComputeStreamIndex();
    });

REGISTER_COMPUTE_TASK_NODE_STREAM_INDEX_GETTER(DeviceType::kCPU, TaskType::kCallbackNotify)
    .SetStreamIndexGetterFn([](DeviceId device_id) -> uint32_t {
      auto* cpu_stream_index_generator = dynamic_cast<CPUStreamIndexGenerator*>(
          Global<IDMgr>::Get()->GetStreamIndexGeneratorManager()->GetGenerator(device_id));
      CHECK_NOTNULL(cpu_stream_index_generator);
      return cpu_stream_index_generator->GenerateIndependentTaskStreamIndex(
          TaskType::kCallbackNotify);
    });

}  // namespace oneflow
