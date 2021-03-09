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
using StreamIndexGetterFn = std::function<uint32_t(DeviceId device_id)>;

class StreamIndexGetterRegistry final {
 public:
  StreamIndexGetterRegistry(DeviceType dev_type, TaskType task_type)
      : dev_task_type_(std::make_pair(dev_type, task_type)) {}
  template<class T>
  StreamIndexGetterRegistry& SetStreamIndexGetterFn(T&& func) {
    // "+ trick": https://stackoverflow.com/a/43843606
    // It is used to convert lambda to a plain function pointer
    // (NOTE: only lambda function without capture can be converted)
    // and thus "GeneratorType" can be deduced
    return SetStreamIndexGetterFnImpl(+func);
  }
  StreamIndexGetterRegistry& SetFn(StreamIndexGetterFn func);

 private:
  std::pair<DeviceType, TaskType> dev_task_type_;

  template<class GeneratorType>
  StreamIndexGetterRegistry& SetStreamIndexGetterFnImpl(uint32_t (*func)(GeneratorType)) {
    auto new_func = [func](DeviceId device_id) -> uint32_t {
      auto* generator = dynamic_cast<GeneratorType>(
          Global<IDMgr>::Get()->GetStreamIndexGeneratorManager()->GetGenerator(device_id));
      CHECK_NOTNULL(generator);
      return func(generator);
    };
    return SetFn(new_func);
  }
};

};  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_STREAM_INDEX_GETTER_REGISTRY_H_
