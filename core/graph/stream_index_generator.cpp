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
#include "oneflow/core/graph/stream_index_generator.h"

namespace oneflow {

StreamIndexGenerator::StreamIndexGenerator() : next_stream_index_(0) {}

StreamIndexGenerator::stream_index_t StreamIndexGenerator::GenerateAnonymous() {
  std::unique_lock<std::mutex> lck(mtx_);
  return next_stream_index_++;
}

StreamIndexGenerator::stream_index_t StreamIndexGenerator::GenerateNamed(const std::string& name) {
  return GenerateNamedRoundRobin(name, 1);
}

StreamIndexGenerator::stream_index_t StreamIndexGenerator::GenerateNamedRoundRobin(
    const std::string& name, size_t size) {
  CHECK_GT(size, 0);
  std::unique_lock<std::mutex> lck(mtx_);
  auto it = name2rr_range_.find(name);
  if (it == name2rr_range_.end()) {
    it = name2rr_range_.emplace(name, RoundRobinRange{next_stream_index_, size}).first;
    next_stream_index_ += size;
  } else {
    CHECK_EQ(it->second.size, size) << name;
  }

  stream_index_t cur_stream_index = it->second.begin;
  if (size > 1) {
    size_t& offset = it->second.offset;
    cur_stream_index += offset++;
    if (offset >= size) { offset = 0; }
  }
  return cur_stream_index;
}

}  // namespace oneflow
