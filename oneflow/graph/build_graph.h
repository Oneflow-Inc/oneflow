#ifndef ONEFLOW_GRAPH_BUILD_GRAPH_H_
#define ONEFLOW_GRAPH_BUILD_GRAPH_H_

#include "graph/task_graph.h"

namespace oneflow {

std::shared_ptr<TaskGraph> BuildTaskGraph(const DLNetConf& dl_net_conf,
                                          const Strategy& strategy_conf,
                                          const IDMap& id_map,
                                          bool need_bp);

} // namespace oneflow

#endif // ONEFLOW_GRAPH_BUILD_GRAPH_H_
