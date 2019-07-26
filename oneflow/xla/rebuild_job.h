#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_REBUILD_JOB_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_REBUILD_JOB_H_

#include "oneflow/core/job/job.pb.h"
#include "oneflow/xla/of2xla/xla_graph.h"

namespace oneflow {

// Rebuild job according to the nodes folded xla graph. In order to rebuild
// the job, We will add several xla launch operators in the job, and remove the
// folded nodes. In xla launch operator, we wll reconstruct the subgraph and
// insert argument nodes if necessary.
void RebuildXlaCompiledJob(const mola::XlaGraph &graph, Job *job);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_REBUILD_JOB_H_
