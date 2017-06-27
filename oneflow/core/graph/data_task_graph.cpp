#include "oneflow/core/graph/data_task_graph.h"

namespace oneflow {

class DataCompTaskNode;

DataTaskGraph::DataTaskGraph(
    const std::string& name,
    const DLNetConf& dl_net_conf,
    const Strategy& strategy_conf,
    bool need_bp) {
  mut_name() = name;
  LogicalGraph logical_gph(dl_net_conf, strategy_conf);
  auto chain_gph = of_make_unique<ChainGraph>(&logical_gph);
  BuildFromChainGph<DataCompTaskNode>(std::move(chain_gph), need_bp);
  BuildExecAndEnrollLbn2Regsts();
}

}
