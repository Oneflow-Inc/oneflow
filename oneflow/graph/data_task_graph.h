#ifndef ONEFLOW_GRAPH_DATA_TASK_GRAPH_H_
#define ONEFLOW_GRAPH_DATA_TASK_GRAPH_H_

#include "graph/task_graph.h"

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

 private:
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_DATA_TASK_GRAPH_H_
