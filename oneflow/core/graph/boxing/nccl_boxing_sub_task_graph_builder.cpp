#include "oneflow/core/graph/boxing/nccl_boxing_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"
#include "oneflow/core/graph/nccl_boxing_task_node.h"

namespace oneflow {

namespace {

bool IsBoxingNotSupported(const Maybe<void>& status) {
  return status.error()->has_boxing_error()
         && status.error()->boxing_error() == BoxingError::kNotSupported;
}

}  // namespace

Maybe<void> NcclBoxingSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, const ParallelDesc& src_parallel_desc,
    const ParallelDesc& dst_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& src_sbp_parallel,
    const SbpParallel& dst_sbp_parallel) const {
  if (!dst_parallel_desc.Equals(src_parallel_desc)) { return Error::BoxingNotSupported(); }
  if (SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)) {
    return Error::BoxingNotSupported();
  }
  if (dst_parallel_desc.device_type() != DeviceType::kGPU) { return Error::BoxingNotSupported(); }
  if (dst_parallel_desc.parallel_num() <= 1) { return Error::BoxingNotSupported(); }
  if (SubTskGphBuilderUtil::IsBoxingP2B(src_sbp_parallel, dst_sbp_parallel)) {
    FOR_RANGE(int64_t, i, 0, dst_parallel_desc.parallel_num()) {
      CompTaskNode* src_node = sorted_src_comp_tasks.at(i);
      CompTaskNode* dst_node = sorted_dst_comp_tasks.at(i);
      NcclBoxingAllReduceTaskNode* nccl_node =
          ctx->task_graph()->NewNode<NcclBoxingAllReduceTaskNode>();
      nccl_node->Init(src_node->machine_id(), src_node->GpuPhyId(), *src_node->parallel_ctx(), lbi);
      Connect<TaskNode>(src_node, ctx->task_graph()->NewEdge(), nccl_node);
      Connect<TaskNode>(nccl_node, ctx->task_graph()->NewEdge(), dst_node);
    }
    return Maybe<void>::Ok();
  } else if (SubTskGphBuilderUtil::IsBoxingP2S(src_sbp_parallel, dst_sbp_parallel)
             && dst_sbp_parallel.split_parallel().axis() == 0
             && logical_blob_desc.shape().At(0) % dst_parallel_desc.parallel_num() == 0) {
    FOR_RANGE(int64_t, i, 0, dst_parallel_desc.parallel_num()) {
      CompTaskNode* src_node = sorted_src_comp_tasks.at(i);
      CompTaskNode* dst_node = sorted_dst_comp_tasks.at(i);
      NcclBoxingReduceScatterTaskNode* nccl_node =
          ctx->task_graph()->NewNode<NcclBoxingReduceScatterTaskNode>();
      nccl_node->Init(src_node->machine_id(), src_node->GpuPhyId(), *src_node->parallel_ctx(), lbi);
      Connect<TaskNode>(src_node, ctx->task_graph()->NewEdge(), nccl_node);
      Connect<TaskNode>(nccl_node, ctx->task_graph()->NewEdge(), dst_node);
    }
    return Maybe<void>::Ok();
  } else if (SubTskGphBuilderUtil::IsBoxingS2B(src_sbp_parallel, dst_sbp_parallel)
             && src_sbp_parallel.split_parallel().axis() == 0
             && logical_blob_desc.shape().At(0) % dst_parallel_desc.parallel_num() == 0) {
    FOR_RANGE(int64_t, i, 0, dst_parallel_desc.parallel_num()) {
      CompTaskNode* src_node = sorted_src_comp_tasks.at(i);
      CompTaskNode* dst_node = sorted_dst_comp_tasks.at(i);
      NcclBoxingAllGatherTaskNode* nccl_node =
          ctx->task_graph()->NewNode<NcclBoxingAllGatherTaskNode>();
      nccl_node->Init(src_node->machine_id(), src_node->GpuPhyId(), *src_node->parallel_ctx(), lbi);
      Connect<TaskNode>(src_node, ctx->task_graph()->NewEdge(), nccl_node);
      Connect<TaskNode>(nccl_node, ctx->task_graph()->NewEdge(), dst_node);
    }
    return Maybe<void>::Ok();
  } else {
    return Error::BoxingNotSupported();
  }
}

}  // namespace oneflow
