#include "oneflow/core/graph/boxing/chain_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"

namespace oneflow {

Maybe<void> ChainSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, const ParallelDesc& src_parallel_desc,
    const ParallelDesc& dst_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& src_sbp_parallel,
    const SbpParallel& dst_sbp_parallel) const {
  for (const auto& builder : builders_) {
    Maybe<void> status = TRY(builder->Build(ctx, sorted_src_comp_tasks, sorted_dst_comp_tasks,
                                            src_parallel_desc, dst_parallel_desc, lbi,
                                            logical_blob_desc, src_sbp_parallel, dst_sbp_parallel));
    if (!status.IsOk() && SubTskGphBuilderUtil::IsErrorBoxingNotSupported(*status.error())) {
      continue;
    } else {
      return status;
    }
  }
  return Error::BoxingNotSupported();
}

}  // namespace oneflow
