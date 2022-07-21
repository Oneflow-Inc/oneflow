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
#ifndef ONEFLOW_CORE_GRAPH_TASK_STREAM_ID_H_
#define ONEFLOW_CORE_GRAPH_TASK_STREAM_ID_H_

#include "oneflow/core/graph/stream_id.h"
#include "oneflow/core/graph/task_stream_index_manager.h"

namespace oneflow {

inline StreamId GenerateComputeTaskStreamId(const DeviceId& device_id) {
  auto stream_index =
      Singleton<TaskStreamIndexManager>::Get()->GetComputeTaskStreamIndex(device_id);
  return StreamId{device_id, stream_index};
}

inline StreamId GenerateComputeTaskStreamId(int64_t rank, DeviceType device_type,
                                            int64_t device_index) {
  DeviceId device_id{static_cast<DeviceId::rank_t>(rank), device_type,
                     static_cast<DeviceId::device_index_t>(device_index)};
  return GenerateComputeTaskStreamId(device_id);
}

inline StreamId GenerateNamedTaskStreamId(const DeviceId& device_id, const std::string& name) {
  auto stream_index =
      Singleton<TaskStreamIndexManager>::Get()->GetNamedTaskStreamIndex(device_id, name);
  return StreamId{device_id, stream_index};
}

inline StreamId GenerateNamedTaskStreamId(int64_t rank, DeviceType device_type,
                                          int64_t device_index, const std::string& name) {
  DeviceId device_id{static_cast<DeviceId::rank_t>(rank), device_type,
                     static_cast<DeviceId::device_index_t>(device_index)};
  return GenerateNamedTaskStreamId(device_id, name);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_TASK_STREAM_ID_H_
