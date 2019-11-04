#ifndef ONEFLOW_CORE_GRAPH_BOXING_NCCL_BOXING_SUB_TASK_GRAPH_BUILDER_H_
#define ONEFLOW_CORE_GRAPH_BOXING_NCCL_BOXING_SUB_TASK_GRAPH_BUILDER_H_

#include "oneflow/core/graph/boxing/sub_task_graph_builder.h"

namespace oneflow {

class NcclBoxingSubTskGphBuilder final : public SubTskGphBuilder {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclBoxingSubTskGphBuilder);
  NcclBoxingSubTskGphBuilder() = default;
  ~NcclBoxingSubTskGphBuilder() override = default;

  Maybe<void> Build(SubTskGphBuilderCtx* ctx,
                    const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
                    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
                    const ParallelDesc& src_parallel_desc, const ParallelDesc& dst_parallel_desc,
                    const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
                    const SbpParallel& src_sbp_parallel,
                    const SbpParallel& dst_sbp_parallel) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_BOXING_NCCL_BOXING_SUB_TASK_GRAPH_BUILDER_H_
