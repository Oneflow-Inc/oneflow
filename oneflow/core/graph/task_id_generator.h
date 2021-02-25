#ifndef ONEFLOW_CORE_GRAPH_TASK_ID_GENERATOR_H_
#define ONEFLOW_CORE_GRAPH_TASK_ID_GENERATOR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/id_util.h"

namespace oneflow {

class TaskIdGenerator final {
 public:
  TaskIdGenerator() = default;
  OF_DISALLOW_COPY_AND_MOVE(TaskIdGenerator);
  ~TaskIdGenerator() = default;

  TaskId Generate(ProcessId process_id, StreamId stream_id);

 private:
  using process_stream_key_t = std::pair<ProcessId, StreamId>;
  HashMap<process_stream_key_t, uint32_t> process_stream2task_index_counter_;
};

inline TaskId TaskIdGenerator::Generate(ProcessId process_id, StreamId stream_id) {
  process_stream_key_t key = std::make_pair(process_id, stream_id);
  uint32_t task_index = process_stream2task_index_counter_[key]++;
  return TaskId(process_id, stream_id, task_index);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_TASK_ID_GENERATOR_H_
