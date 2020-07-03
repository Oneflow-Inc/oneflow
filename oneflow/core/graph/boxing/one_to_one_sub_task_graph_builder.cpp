#include "oneflow/core/graph/boxing/one_to_one_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"

namespace oneflow {

Maybe<void> OneToOneSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, const ParallelDesc& src_parallel_desc,
    const ParallelDesc& dst_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& src_sbp_parallel,
    const SbpParallel& dst_sbp_parallel) const {
  if ((src_parallel_desc.parallel_num() == 1 && dst_parallel_desc.parallel_num() == 1)
      || (src_parallel_desc.parallel_num() == dst_parallel_desc.parallel_num()
          && src_sbp_parallel == dst_sbp_parallel)) {
    for (int64_t i = 0; i < src_parallel_desc.parallel_num(); ++i) {
      CompTaskNode* src_node = sorted_src_comp_tasks.at(i);
      CompTaskNode* dst_node = sorted_dst_comp_tasks.at(i);
      // TODO(liujuncheng): use lbi
      TaskNode* proxy = ctx->GetProxyNode(src_node, src_node->MemZoneId121(),
                                          dst_node->machine_id(), dst_node->MemZoneId121());
      Connect<TaskNode>(proxy, ctx->task_graph()->NewEdge(), dst_node);
    }
    return Maybe<void>::Ok();
  } else {
    return Error::BoxingNotSupported();
  }
}

}  // namespace oneflow
