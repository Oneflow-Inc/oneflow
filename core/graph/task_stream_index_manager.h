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
#ifndef ONEFLOW_CORE_GRAPH_TASK_STREAM_INDEX_MANAGER_H_
#define ONEFLOW_CORE_GRAPH_TASK_STREAM_INDEX_MANAGER_H_

#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/graph/stream_index_generator.h"

namespace oneflow {

class TaskStreamIndexManager final {
 public:
  using stream_index_t = StreamId::stream_index_t;

  OF_DISALLOW_COPY_AND_MOVE(TaskStreamIndexManager);
  TaskStreamIndexManager() = default;
  ~TaskStreamIndexManager() = default;

  StreamIndexGenerator* GetGenerator(const DeviceId& device_id);
  stream_index_t GetTaskStreamIndex(TaskType task_type, const DeviceId& device_id);
  stream_index_t GetComputeTaskStreamIndex(const DeviceId& device_id);
  stream_index_t GetNamedTaskStreamIndex(const DeviceId& device_id, const std::string& name);

 private:
  HashMap<DeviceId, std::unique_ptr<StreamIndexGenerator>> generators_;
  std::mutex mtx_;
};

class TaskStreamIndexGetterRegistry final {
 public:
  using key_t = std::pair<DeviceType, TaskType>;
  using stream_index_getter = std::function<StreamId::stream_index_t(StreamIndexGenerator*)>;
  using map_t = HashMap<key_t, stream_index_getter>;

  struct GetterRegister {
    GetterRegister(DeviceType device_type, TaskType task_type, const stream_index_getter& getter) {
      TaskStreamIndexGetterRegistry::Instance().Register(std::make_pair(device_type, task_type),
                                                         getter);
    }
  };

  static TaskStreamIndexGetterRegistry& Instance() {
    static TaskStreamIndexGetterRegistry registry;
    return registry;
  }

  OF_DISALLOW_COPY_AND_MOVE(TaskStreamIndexGetterRegistry);
  ~TaskStreamIndexGetterRegistry() = default;

  void Register(const key_t& key, const stream_index_getter& getter);
  Maybe<StreamId::stream_index_t> Dispatch(DeviceType device_type, TaskType task_type,
                                           StreamIndexGenerator* generator);

 private:
  TaskStreamIndexGetterRegistry() = default;
  map_t stream_index_getter_map_;
};

StreamId::stream_index_t GenerateComputeTaskStreamIndex(DeviceType device_type,
                                                        StreamIndexGenerator* generator);

}  // namespace oneflow

#define REGISTER_TASK_STREAM_INDEX_GETTER(device_type, task_type, getter) \
  static auto OF_PP_CAT(g_stream_index_getter_register_, __COUNTER__) =   \
      ::oneflow::TaskStreamIndexGetterRegistry::GetterRegister(device_type, task_type, getter)

#define REGISTER_NAMED_TASK_STREAM_INDEX_GETTER(device_type, task_type, name)                    \
  REGISTER_TASK_STREAM_INDEX_GETTER(                                                             \
      device_type, task_type, ([](StreamIndexGenerator* generator) -> StreamId::stream_index_t { \
        return generator->GenerateNamed(name);                                                   \
      }));

#define REGISTER_INDEPENDENT_TASK_STREAM_INDEX_GETTER(task_type)         \
  REGISTER_TASK_STREAM_INDEX_GETTER(                                     \
      DeviceType::kCPU, task_type,                                       \
      ([](StreamIndexGenerator* generator) -> StreamId::stream_index_t { \
        return generator->GenerateAnonymous();                           \
      }));

#define REGISTER_TICK_TASK_STREAM_INDEX_GETTER(task_type)                \
  REGISTER_TASK_STREAM_INDEX_GETTER(                                     \
      DeviceType::kCPU, task_type,                                       \
      ([](StreamIndexGenerator* generator) -> StreamId::stream_index_t { \
        return generator->GenerateNamed("TICK");                         \
      }));

#define REGISTER_DEVICE_COMP_TASK_STREAM_INDEX_GETTER(device_type, task_type)                    \
  REGISTER_TASK_STREAM_INDEX_GETTER(                                                             \
      device_type, task_type, ([](StreamIndexGenerator* generator) -> StreamId::stream_index_t { \
        return GenerateComputeTaskStreamIndex(device_type, generator);                           \
      }));

#define REGISTER_COMP_TASK_STREAM_INDEX_GETTER(task_type)                                          \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_DEVICE_COMP_TASK_STREAM_INDEX_GETTER, DEVICE_TYPE_SEQ, \
                                   (task_type))

#endif  // ONEFLOW_CORE_GRAPH_TASK_STREAM_INDEX_MANAGER_H_
