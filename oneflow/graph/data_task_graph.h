#ifndef ONEFLOW_GRAPH_DATA_TASK_GRAPH_H_
#define ONEFLOW_GRAPH_DATA_TASK_GRAPH_H_

#include "graph/task_graph.h"

namespace oneflow {

class DataTaskGraph final : public TaskGraph {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DataTaskGraph);
  DataTaskGraph() = delete;
  ~DataTaskGraph() = default;
  
  DataTaskGraph(const DLNetConf& dl_net_conf,
                const Strategy& strategy_conf,
                bool need_bp) {
    LogicalGraph logical_gph(dl_net_conf, strategy_conf);
    logical_gph.ToDotFile(LogDir() + "/logical_graph.dot");
    auto chain_gph = of_make_unique<ChainGraph> (&logical_gph);
    BuildFromChainGph(std::move(chain_gph), need_bp);
  }

  CompTaskNodeMemFunc Func4FwBuildExecAndProducedRegsts() const override {
    return &CompTaskNode::DataFwBuildExecAndProducedRegsts;
  }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_DATA_TASK_GRAPH_H_
