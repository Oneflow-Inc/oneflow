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

StreamIndexGenerator::stream_index_t StreamIndexGenerator::operator()() {
  return next_stream_index_.fetch_add(1, std::memory_order_relaxed);
}

StreamIndexGenerator::stream_index_t StreamIndexGenerator::operator()(const std::string& name) {
  return (*this)(name, 1);
}

StreamIndexGenerator::stream_index_t StreamIndexGenerator::operator()(const std::string& name,
                                                                      size_t num) {
  CHECK_GT(num, 0);
  std::unique_lock<std::mutex> lck1(named_rr_range_mutex_);
  auto range_it = name2round_robin_range_.find(name);
  if (range_it == name2round_robin_range_.end()) {
    stream_index_t begin_stream_index =
        next_stream_index_.fetch_add(num, std::memory_order_relaxed);
    range_it = name2round_robin_range_.emplace(name, std::make_pair(begin_stream_index, num)).first;
  } else {
    CHECK_EQ(range_it->second.second, num) << name;
  }

  stream_index_t cur_stream_index = range_it->second.first;
  if (num > 1) {
    std::unique_lock<std::mutex> lck2(named_rr_offset_mutex_);
    auto offset_it = name2round_robine_offset.find(name);
    if (offset_it == name2round_robine_offset.end()) {
      offset_it = name2round_robine_offset.emplace(name, 0).first;
    }
    cur_stream_index += offset_it->second++;
    if (offset_it->second > range_it->second.second) { offset_it->second = 0; }
  }
  return cur_stream_index;
}

}  // namespace oneflow
