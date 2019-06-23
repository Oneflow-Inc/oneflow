#ifndef ONEFLOW_CORE_COMPILER_OF2XLA_REBUILD_JOB_H_
#define ONEFLOW_CORE_COMPILER_OF2XLA_REBUILD_JOB_H_

#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/compiler/of2xla/xla_graph.h"

namespace oneflow {

// Rebuild Job according to the nodes folded xla graph.
// In order to rebuild the job, We will add new xla launch nodes in the job,
// and move that folded nodes into their xla launch nodes. Beyond that we
// also create the argument nodes for all xla launch nodes
void RebuildXlaCompiledJob(const mola::XlaGraph &graph, Job *job);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMPILER_OF2XLA_REBUILD_JOB_H_
