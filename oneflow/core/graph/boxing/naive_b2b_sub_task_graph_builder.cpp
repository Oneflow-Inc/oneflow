#include "oneflow/core/graph/boxing/naive_b2b_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"

namespace oneflow {

Maybe<void> NaiveB2BSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, const ParallelDesc& src_parallel_desc,
    const ParallelDesc& dst_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& src_sbp_parallel,
    const SbpParallel& dst_sbp_parallel) const {
  if ((src_parallel_desc.parallel_num() == 1 || src_sbp_parallel.has_broadcast_parallel())
      && (dst_parallel_desc.parallel_num() == 1 || dst_sbp_parallel.has_broadcast_parallel())) {
    std::vector<CompTaskNode*> nearest_src_comp_tasks;
    for (CompTaskNode* dst_node : sorted_dst_comp_tasks) {
      CompTaskNode* nearest_src_node =
          SubTskGphBuilderUtil::FindNearestNode(sorted_src_comp_tasks, dst_node);
      CHECK_NOTNULL(nearest_src_node);
      TaskNode* proxy = ctx->GetProxyNode(nearest_src_node, nearest_src_node->MemZoneId121(),
                                          dst_node->machine_id(), dst_node->MemZoneId121());
      Connect<TaskNode>(proxy, ctx->task_graph()->NewEdge(), dst_node);
    }
    return Maybe<void>::Ok();
  } else {
    return Error::BoxingNotSupported();
  }
}

}  // namespace oneflow
