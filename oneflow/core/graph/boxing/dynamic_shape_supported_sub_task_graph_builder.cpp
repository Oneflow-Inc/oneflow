#include "oneflow/core/graph/boxing/dynamic_shape_supported_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing_concat_task_node.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"

namespace oneflow {

SubTskGphBuilderStatus DynamicShapeSupportedSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, const ParallelDesc& src_parallel_desc,
    const ParallelDesc& dst_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& src_sbp_parallel,
    const SbpParallel& dst_sbp_parallel) const {
  if (!logical_blob_desc.has_dim0_valid_num_field()) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  if (src_parallel_desc.parallel_num() < 2) { return SubTskGphBuilderStatus::MakeStatusError(); }
  if (!src_sbp_parallel.has_split_parallel()) { return SubTskGphBuilderStatus::MakeStatusError(); }
  if (dst_parallel_desc.parallel_num() != 1 && !dst_sbp_parallel.has_broadcast_parallel()) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  FOR_RANGE(int64_t, dst_id, 0, dst_parallel_desc.parallel_num()) {
    TaskNode* dst_node = sorted_dst_comp_tasks.at(dst_id);
    BoxingConcatTaskNode* concat_task_node = ctx->task_graph()->NewNode<BoxingConcatTaskNode>();
    concat_task_node->Init(lbi, dst_node->machine_id(), dst_node->thrd_id(),
                           src_sbp_parallel.split_parallel().axis());
    FOR_RANGE(int64_t, src_id, 0, src_parallel_desc.parallel_num()) {
      TaskNode* src_node = sorted_src_comp_tasks.at(src_id);
      TaskNode* proxy = ctx->GetProxyNode(src_node, GetDefaultMemCase(src_node),
                                          dst_node->machine_id(), GetDefaultMemCase(dst_node));
      concat_task_node->ConnectToSrc(proxy, ctx->task_graph()->NewEdge());
    }
    Connect<TaskNode>(concat_task_node, ctx->task_graph()->NewEdge(), dst_node);
  }
  return SubTskGphBuilderStatus::MakeStatusOK();
}

}  // namespace oneflow
