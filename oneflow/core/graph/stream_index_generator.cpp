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

StreamIndexGenerator::stream_index_t StreamIndexGenerator::Generate() {
  std::unique_lock<std::mutex> lck(mtx_);
  return next_stream_index_++;
}

StreamIndexGenerator::stream_index_t StreamIndexGenerator::Generate(const std::string& name) {
  return Generate(name, 1);
}

StreamIndexGenerator::stream_index_t StreamIndexGenerator::Generate(const std::string& name,
                                                                    size_t num) {
  CHECK_GT(num, 0);
  std::unique_lock<std::mutex> lck(mtx_);
  auto it = name2round_robin_tup_.find(name);
  if (it == name2round_robin_tup_.end()) {
    // tuple of (begin_stream_index, num, offset)
    auto tup = std::make_tuple(next_stream_index_, num, 0);
    it = name2round_robin_tup_.emplace(name, std::move(tup)).first;
    next_stream_index_ += num;
  } else {
    CHECK_EQ(std::get<1>(it->second), num) << name;
  }

  stream_index_t cur_stream_index = std::get<0>(it->second);
  if (num > 1) {
    size_t& offset = std::get<2>(it->second);
    cur_stream_index += offset++;
    if (offset > num) { offset = 0; }
  }
  return cur_stream_index;
}

}  // namespace oneflow
