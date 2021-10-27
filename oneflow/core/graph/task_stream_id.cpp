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
#include "oneflow/core/graph/task_stream_id.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/stream/stream_index_generator.h"

namespace oneflow {

StreamId GenerateComputeStreamId(const DeviceId& device_id) {
  auto* stream_index_generator =
      Global<IDMgr>::Get()->GetStreamIndexGeneratorManager()->GetOrCreateGenerator(device_id);
  StreamId::index_t stream_index = 0;
  if (device_id.device_type() == DeviceType::kCPU) {
    size_t cpu_device_num = Global<ResourceDesc, ForSession>::Get()->CpuDeviceNum();
    stream_index = stream_index_generator->GenerateStreamIndex("cpu_compute", cpu_device_num);
  } else {
    stream_index = stream_index_generator->GenerateStreamIndex("compute");
  }
  return StreamId{device_id, stream_index};
}

StreamId GenerateComputeStreamId(int64_t node_index, DeviceType device_type, int64_t device_index) {
  DeviceId device_id{static_cast<DeviceId::index_t>(node_index), device_type,
                     static_cast<DeviceId::index_t>(device_index)};
  return GenerateComputeStreamId(device_id);
}

}  // namespace oneflow
