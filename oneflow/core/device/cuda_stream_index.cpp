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
#include "oneflow/core/device/cuda_stream_index.h"

namespace oneflow {

CudaStreamIndexGenerator::CudaStreamIndexGenerator() { next_stream_index_ = kD2H + 1; }

CudaStreamIndexGenerator::~CudaStreamIndexGenerator() = default;

StreamIndexGenerator::stream_index_t CudaStreamIndexGenerator::GenerateNamedStreamIndex(
    const std::string& name) {
  std::lock_guard<std::mutex> lock(named_stream_index_mutex_);
  auto it = named_stream_index_.find(name);
  if (it == named_stream_index_.end()) {
    stream_index_t index = next_stream_index_;
    next_stream_index_ += 1;
    named_stream_index_.emplace(name, index);
    return index;
  } else {
    return it->second;
  }
}

bool CudaStreamIndexGenerator::IsNamedStreamIndex(const std::string& name, stream_index_t index) {
  std::lock_guard<std::mutex> lock(named_stream_index_mutex_);
  auto it = named_stream_index_.find(name);
  if (it == named_stream_index_.end()) {
    return false;
  } else {
    return it->second == index;
  }
}

REGISTER_STREAM_INDEX_GENERATOR(DeviceType::kGPU, CudaStreamIndexGenerator);

}  // namespace oneflow
