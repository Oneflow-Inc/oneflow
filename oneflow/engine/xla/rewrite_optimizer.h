#ifndef ONEFLOW_ENGINE_XLA_REWRITE_OPTIMIZER_GRAPH_H_
#define ONEFLOW_ENGINE_XLA_REWRITE_OPTIMIZER_GRAPH_H_

#include "oneflow/core/job/job.pb.h"
#include "oneflow/engine/xla/of2xla/xla_graph.h"

namespace oneflow {

// Rewrite model update operator to optimizer graph
void RewriteOptimizerGraph(const mla::XlaGraph &graph, Job *job);

}  // namespace oneflow

#endif  // ONEFLOW_ENGINE_XLA_REWRITE_OPTIMIZER_GRAPH_H_
