#include "oneflow/core/schedule/utilization_graph.h"

namespace oneflow {
namespace schedule {

void UtilizationGraph::ForEachUtilization(
    const std::function<void(Utilization*)>& cb) const {
  cb(const_cast<ComputationUtilization*>(&computation()));
  cb(const_cast<MemoryUtilization*>(&memory()));
  dev_computation_mgr().ForEach(cb);
  stream_mgr().ForEach(cb);
  task_mgr().ForEach(cb);
  task_stream_mgr().ForEach(cb);
  dev_memory_mgr().ForEach(cb);
  regst_desc_mgr().ForEach(cb);
  regst_mgr().ForEach(cb);
}

}  // namespace schedule
}  // namespace oneflow
