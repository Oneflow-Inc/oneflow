#ifndef ONEFLOW_CORE_GRAPH_DATA_TASK_GRAPH_H_
#define ONEFLOW_CORE_GRAPH_DATA_TASK_GRAPH_H_

#include "oneflow/core/graph/task_graph.h"

namespace oneflow {

class DataTaskGraph final : public TaskGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataTaskGraph);
  DataTaskGraph() = delete;
  ~DataTaskGraph() = default;
  
  DataTaskGraph(const std::string& name,
                const DLNetConf& dl_net_conf,
                const Strategy& strategy_conf,
                bool need_bp);
  
  const char* TypeName() const override { return "DataTaskGraph"; }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_CORE_GRAPH_DATA_TASK_GRAPH_H_
