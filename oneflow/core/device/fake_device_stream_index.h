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
#ifndef ONEFLOW_CORE_DEVICE_FAKE_DEVICE_STREAM_INDEX_H_
#define ONEFLOW_CORE_DEVICE_FAKE_DEVICE_STREAM_INDEX_H_

#include "oneflow/core/device/stream_index.h"
#include "oneflow/core/graph/task_node.h"

namespace oneflow {

class FakeDeviceStreamIndexGenerator final : public StreamIndexGenerator {
 public:
  FakeDeviceStreamIndexGenerator();
  OF_DISALLOW_COPY_AND_MOVE(FakeDeviceStreamIndexGenerator);
  ~FakeDeviceStreamIndexGenerator() = default;

  stream_index_t GenerateComputeStreamIndex() override { return 0; }
  stream_index_t GenerateH2DStreamIndex() override { return 1; }
  stream_index_t GenerateD2HStreamIndex() override { return 2; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_FAKE_DEVICE_STREAM_INDEX_H_
