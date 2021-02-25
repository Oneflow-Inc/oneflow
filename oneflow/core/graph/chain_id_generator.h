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
