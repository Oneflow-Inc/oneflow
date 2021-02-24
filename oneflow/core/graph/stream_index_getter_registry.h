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
#ifndef ONEFLOW_CORE_GRAPH_STREAM_INDEX_GETTER_REGISTRY_H_
#define ONEFLOW_CORE_GRAPH_STREAM_INDEX_GETTER_REGISTRY_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/graph/task_node.h"
#include "oneflow/core/common/device_type.pb.h"

namespace oneflow {

class CompTaskNode;
using StreamIndexGetterFn = std::function<int64_t(const CompTaskNode*, std::function<uint32_t(int task_type)>, std::function<uint32_t(const TaskNode*)>)>;

class StreamIndexGetterRegistry final {
 public:
  StreamIndexGetterRegistry(DeviceType dev_type, TaskType task_type)
      : dev_task_type_(std::make_pair(dev_type, task_type)) {}
  StreamIndexGetterRegistry& SetStreamIndexGetterFn(StreamIndexGetterFn func);

 private:
  std::pair<DeviceType, TaskType> dev_task_type_;
};

};  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_STREAM_INDEX_GETTER_REGISTRY_H_
