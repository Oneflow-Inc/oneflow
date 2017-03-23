#ifndef ONEFLOW_GRAPH_BUILD_GRAPH_H_
#define ONEFLOW_GRAPH_BUILD_GRAPH_H_

#include "graph/task_graph.h"

namespace oneflow {

std::shared_ptr<TaskGraph> BuildTaskGraph();

} // namespace oneflow

#endif // ONEFLOW_GRAPH_BUILD_GRAPH_H_
