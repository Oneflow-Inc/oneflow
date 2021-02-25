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
#ifndef ONEFLOW_CORE_GRAPH_CHAIN_ID_GENERATOR_H_
#define ONEFLOW_CORE_GRAPH_CHAIN_ID_GENERATOR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/id_util.h"

namespace oneflow {

using chain_id_t = TaskId;

class ChainIdGenerator final {
 public:
  ChainIdGenerator() = default;
  OF_DISALLOW_COPY_AND_MOVE(ChainIdGenerator);
  ~ChainIdGenerator() = default;

  chain_id_t Generate(uint64_t global_stream_index);

 private:
  using process_stream_key_t = std::pair<ProcessId, StreamId>;
  HashMap<process_stream_key_t, uint32_t> process_stream2chain_index_counter_;
};

inline chain_id_t ChainIdGenerator::Generate(uint64_t global_stream_index) {
  chain_id_t chain_id_with_empty_chain_index{global_stream_index, 0};
  process_stream_key_t key = std::make_pair(chain_id_with_empty_chain_index.process_id(),
                                            chain_id_with_empty_chain_index.stream_id());
  uint32_t task_index = process_stream2chain_index_counter_[key]++;
  return chain_id_t{global_stream_index, task_index};
}

inline int64_t SerializeChainIdToInt64(chain_id_t chain_id) {
  return SerializeTaskIdToInt64(chain_id);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_CHAIN_ID_GENERATOR_H_
