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
    LogicalGraph logical_gph(dl_net_conf, strategy_conf, LogDir() + "/logical_graph.dot");
    auto chain_gph = of_make_unique<ChainGraph> (&logical_gph, LogDir() + "/data_chain_graph.dot");
    BuildFromChainGph(std::move(chain_gph), need_bp, LogDir() + "/data_");
  }

  CompTaskNodeMemFunc Func4FwBuildExecAndEnrollLbn2Regsts() const override {
    return &CompTaskNode::DataFwBuildExecAndEnrollLbn2Regsts;
  }
  CompTaskNodeMemFunc Func4FwInferShape4LbnInProducedRegsts() const override {
    return &CompTaskNode::DataFwInferShape4LbnInProducedRegsts;
  }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_GRAPH_DATA_TASK_GRAPH_H_
