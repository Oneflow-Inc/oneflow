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

class StreamIndexGenerator final {
 public:
  using stream_index_t = StreamId::stream_index_t;

  StreamIndexGenerator();
  OF_DISALLOW_COPY_AND_MOVE(StreamIndexGenerator);
  ~StreamIndexGenerator() = default;

  stream_index_t GenerateStreamIndex();
  stream_index_t GenerateStreamIndex(const std::string& name, size_t num = 1);

 private:
  std::atomic<stream_index_t> next_stream_index_;
  HashMap<std::string, std::pair<stream_index_t, size_t>> name2round_robin_range_;
  HashMap<std::string, int> name2round_robine_offset;
  std::mutex named_rr_range_mutex_;
  std::mutex named_rr_offset_mutex_;
};

class StreamIndexGeneratorManager final {
 public:
  StreamIndexGeneratorManager() = default;
  OF_DISALLOW_COPY_AND_MOVE(StreamIndexGeneratorManager);
  ~StreamIndexGeneratorManager() = default;

  StreamIndexGenerator* GetOrCreateGenerator(const DeviceId& device_id);

 private:
  HashMap<DeviceId, std::unique_ptr<StreamIndexGenerator>> generators_;
  std::mutex mtx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_STREAM_INDEX_H_
