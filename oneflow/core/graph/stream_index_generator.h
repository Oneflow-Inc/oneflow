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
#ifndef ONEFLOW_CORE_GRAPH_STREAM_INDEX_GENERATOR_H_
#define ONEFLOW_CORE_GRAPH_STREAM_INDEX_GENERATOR_H_

#include "oneflow/core/graph/stream_id.h"

namespace oneflow {

class StreamIndexGenerator final {
 public:
  using stream_index_t = StreamId::stream_index_t;

  StreamIndexGenerator();
  OF_DISALLOW_COPY_AND_MOVE(StreamIndexGenerator);
  ~StreamIndexGenerator() = default;

  stream_index_t GenerateAnonymous();
  stream_index_t GenerateNamed(const std::string& name);
  stream_index_t GenerateNamedRoundRobin(const std::string& name, size_t size);

 private:
  struct RoundRobinRange {
    RoundRobinRange(stream_index_t begin, size_t size) : begin(begin), size(size), offset(0) {}
    stream_index_t begin;
    size_t size;
    size_t offset;
  };

  stream_index_t next_stream_index_;
  HashMap<std::string, RoundRobinRange> name2rr_range_;
  std::mutex mtx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_STREAM_INDEX_GENERATOR_H_
