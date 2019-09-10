#ifndef ONEFLOW_CORE_COMPILER_REWRITE_OPTIMIZER_GRAPH_H_
#define ONEFLOW_CORE_COMPILER_REWRITE_OPTIMIZER_GRAPH_H_

#include "oneflow/core/job/job.pb.h"
#include "oneflow/xla/of2xla/xla_graph.h"

namespace oneflow {

// Rewrite model update operator to optimizer graph
void RewriteOptimizerGraph(const mola::XlaGraph &graph, Job *job);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_REWRITE_OPTIMIZER_GRAPH_H_
