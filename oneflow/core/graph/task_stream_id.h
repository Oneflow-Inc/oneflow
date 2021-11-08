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

#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/graph/stream_index_generator.h"

namespace oneflow {

class StreamIndexGeneratorManager final {
 public:
  StreamIndexGeneratorManager() = default;
  OF_DISALLOW_COPY_AND_MOVE(StreamIndexGeneratorManager);
  ~StreamIndexGeneratorManager() = default;

  StreamIndexGenerator* GetGenerator(const DeviceId& device_id);

 private:
  HashMap<DeviceId, std::unique_ptr<StreamIndexGenerator>> generators_;
  std::mutex mtx_;
};

class TaskStreamIndexRegistry final {
 public:
  using stream_index_getter_fn = std::function<StreamId::stream_index_t(const DeviceId&)>;
  using stream_index_gen_fn = std::function<StreamId::stream_index_t(StreamIndexGenerator*)>;
  using key_t = std::pair<DeviceType, TaskType>;
  using map_t = HashMap<key_t, stream_index_getter_fn>;

  struct GetterRegister {
    GetterRegister(DeviceType device_type, TaskType task_type, const stream_index_gen_fn& gen) {
      auto getter = [gen](const DeviceId& device_id) -> StreamId::stream_index_t {
        auto* generator = Global<StreamIndexGeneratorManager>::Get()->GetGenerator(device_id);
        return gen(generator);
      };
      auto key = std::make_pair(device_type, task_type);
      TaskStreamIndexRegistry::Instance().RegisterGetter(key, getter);
    }
  };

  static TaskStreamIndexRegistry& Instance() {
    static TaskStreamIndexRegistry factory;
    return factory;
  }
  OF_DISALLOW_COPY_AND_MOVE(TaskStreamIndexRegistry);
  ~TaskStreamIndexRegistry() = default;

  void RegisterGetter(const key_t& key, const stream_index_getter_fn& getter);
  Maybe<StreamId::stream_index_t> GetStreamIndex(TaskType task_type, const DeviceId& device_id);

 private:
  TaskStreamIndexRegistry() = default;
  map_t stream_index_getter_map_;
};

Maybe<StreamId::stream_index_t> GetTaskStreamIndex(TaskType task_type, const DeviceId& device_id);

StreamId::stream_index_t GetComputeTaskStreamIndex(DeviceType device_type,
                                                   StreamIndexGenerator* generator);

StreamId GenerateComputeTaskStreamId(const DeviceId& device_id);
StreamId GenerateComputeTaskStreamId(int64_t rank, DeviceType device_type, int64_t device_index);
StreamId GenerateNamedTaskStreamId(const DeviceId& device_id, const std::string& name);
StreamId GenerateNamedTaskStreamId(int64_t rank, DeviceType device_type, int64_t device_index,
                                   const std::string& name);

}  // namespace oneflow

#define REGISTER_TASK_STREAM_INDEX_GETTER(device_type, task_type, getter) \
  static auto OF_PP_CAT(g_stream_index_getter_register_, __COUNTER__) =   \
      ::oneflow::TaskStreamIndexRegistry::GetterRegister(device_type, task_type, getter)

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
        return generator->GenerateNamed("tick");                         \
      }));

#define REGISTER_DEVICE_COMP_TASK_STREAM_INDEX_GETTER(device_type, task_type)                    \
  REGISTER_TASK_STREAM_INDEX_GETTER(                                                             \
      device_type, task_type, ([](StreamIndexGenerator* generator) -> StreamId::stream_index_t { \
        return GetComputeTaskStreamIndex(device_type, generator);                                \
      }));

#define REGISTER_COMP_TASK_STREAM_INDEX_GETTER(task_type)                                          \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_DEVICE_COMP_TASK_STREAM_INDEX_GETTER, DEVICE_TYPE_SEQ, \
                                   (task_type))

#endif  // ONEFLOW_CORE_GRAPH_TASK_STREAM_ID_H_
