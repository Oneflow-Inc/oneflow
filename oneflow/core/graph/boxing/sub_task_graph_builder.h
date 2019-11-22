#ifndef ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_H_
#define ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_context.h"

namespace oneflow {

class SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SubTskGphBuilder);
  SubTskGphBuilder() = default;
  virtual ~SubTskGphBuilder() = default;

  virtual Maybe<void> Build(SubTskGphBuilderCtx* ctx,
                            const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
                            const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
                            const ParallelDesc& src_parallel_desc,
                            const ParallelDesc& dst_parallel_desc, const LogicalBlobId& lbi,
                            const BlobDesc& logical_blob_desc, const SbpParallel& src_sbp_parallel,
                            const SbpParallel& dst_sbp_parallel) const = 0;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_SUB_TASK_GRAPH_BUILDER_H_
