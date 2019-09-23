#include "oneflow/core/graph/boxing/nccl_boxing_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"
#include "oneflow/core/graph/nccl_boxing_task_node.h"

namespace oneflow {

namespace {

template<typename DerivedNodeType>
void NcclBldSubTskGph(SubTskGphBuilderCtx* ctx,
                      const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
                      const std::vector<CompTaskNode*>& sorted_dst_comp_tasks,
                      const LogicalBlobId& lbi) {
  const int64_t parallel_num = sorted_dst_comp_tasks.size();
  const int64_t rank_set_id = NewNodeId() << 32;
  FOR_RANGE(int64_t, i, 0, parallel_num) {
    ParallelContext parallel_ctx{};
    parallel_ctx.set_parallel_id(i);
    parallel_ctx.set_parallel_num(parallel_num);
    parallel_ctx.mutable_rank_ctx()->set_rank_id(i);
    parallel_ctx.mutable_rank_ctx()->set_rank_num(parallel_num);
    parallel_ctx.mutable_rank_ctx()->set_rank_set_id(rank_set_id);
    CompTaskNode* src_node = sorted_src_comp_tasks.at(i);
    CompTaskNode* dst_node = sorted_dst_comp_tasks.at(i);
    DerivedNodeType* nccl_node = ctx->task_graph()->NewNode<DerivedNodeType>();
    nccl_node->Init(src_node->machine_id(), src_node->GpuPhyId(), parallel_ctx, lbi);
    Connect<TaskNode>(src_node, ctx->task_graph()->NewEdge(), nccl_node);
    Connect<TaskNode>(nccl_node, ctx->task_graph()->NewEdge(), dst_node);
  }
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
  if (logical_blob_desc.shape().elem_cnt() < 1024) { return Error::BoxingNotSupported(); }
  if (dst_parallel_desc.device_type() != DeviceType::kGPU) { return Error::BoxingNotSupported(); }
  if (dst_parallel_desc.parallel_num() <= 1) { return Error::BoxingNotSupported(); }
  if (SubTskGphBuilderUtil::IsBoxingP2B(src_sbp_parallel, dst_sbp_parallel)) {
    NcclBldSubTskGph<NcclBoxingAllReduceTaskNode>(ctx, sorted_src_comp_tasks, sorted_dst_comp_tasks,
                                                  lbi);
    return Maybe<void>::Ok();
  } else if (SubTskGphBuilderUtil::IsBoxingP2S(src_sbp_parallel, dst_sbp_parallel)
             && dst_sbp_parallel.split_parallel().axis() == 0
             && logical_blob_desc.shape().At(0) % dst_parallel_desc.parallel_num() == 0) {
    NcclBldSubTskGph<NcclBoxingReduceScatterTaskNode>(ctx, sorted_src_comp_tasks,
                                                      sorted_dst_comp_tasks, lbi);
    return Maybe<void>::Ok();
  } else if (SubTskGphBuilderUtil::IsBoxingS2B(src_sbp_parallel, dst_sbp_parallel)
             && src_sbp_parallel.split_parallel().axis() == 0
             && logical_blob_desc.shape().At(0) % dst_parallel_desc.parallel_num() == 0) {
    NcclBldSubTskGph<NcclBoxingAllGatherTaskNode>(ctx, sorted_src_comp_tasks, sorted_dst_comp_tasks,
                                                  lbi);
    return Maybe<void>::Ok();
  } else {
    return Error::BoxingNotSupported();
  }
}

}  // namespace oneflow
