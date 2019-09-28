#ifndef ONEFLOW_ENGINE_PASS_REBUILD_CLUSTERED_JOB_H_
#define ONEFLOW_ENGINE_PASS_REBUILD_CLUSTERED_JOB_H_

#include "oneflow/core/job/job.pb.h"
#include "oneflow/engine/xla/of2xla/xla_graph.h"

namespace oneflow {

// Rebuild job according to the nodes folded xla graph. In order to rebuild
// the job, We will add several xla launch operators in the job, and remove the
// folded nodes. In xla launch operator, we wll reconstruct the subgraph and
// insert argument nodes if necessary.
void RebuildClusteredJob(const mla::XlaGraph &graph, Job *job);

}  // namespace oneflow

#endif  // ONEFLOW_ENGINE_PASS_REBUILD_CLUSTERED_JOB_H_
