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
#ifndef ONEFLOW_CORE_DEVICE_STREAM_INDEX_H_
#define ONEFLOW_CORE_DEVICE_STREAM_INDEX_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/common/id_util.h"

namespace oneflow {

using stream_index_t = uint32_t;

class StreamIndexGenerator {
 public:
  virtual stream_index_t GenerateComputeStreamIndex() = 0;
  virtual stream_index_t GenerateH2DStreamIndex() = 0;
  virtual stream_index_t GenerateD2HStreamIndex() = 0;

  virtual bool IsComputeStreamIndex(stream_index_t index) const = 0;
  virtual bool IsH2DStreamIndex(stream_index_t index) const = 0;
  virtual bool IsD2HStreamIndex(stream_index_t index) const = 0;
};

class StreamIndexGeneratorManager final {
 public:
  StreamIndexGeneratorManager() = default;
  OF_DISALLOW_COPY_AND_MOVE(StreamIndexGeneratorManager);
  ~StreamIndexGeneratorManager() = default;

  StreamIndexGenerator* GetGenerator(ProcessId process_id, DeviceId device_id);

 private:
  using generator_key_t = std::pair<ProcessId, DeviceId>;
  HashMap<generator_key_t, std::unique_ptr<StreamIndexGenerator>> generators_;
};

inline StreamIndexGenerator* StreamIndexGeneratorManager::GetGenerator(ProcessId process_id,
                                                                       DeviceId device_id) {
  generator_key_t key = std::make_pair(process_id, device_id);
  auto iter = generators_.find(key);
  if (iter == generators_.end()) {
    iter = generators_.emplace(key, nullptr).first;
    iter->second.reset(NewObj<int, StreamIndexGenerator>(device_id.device_type()));
  }
  return iter->second.get();
}

#define REGISTER_STREAM_INDEX_GENERATOR(device_type_v, stream_index_generator_class) \
  REGISTER_CLASS(int, device_type_v, StreamIndexGenerator, stream_index_generator_class)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_STREAM_INDEX_H_
