#include "oneflow/core/graph/boxing/local_ring_sub_task_graph_builder.h"
#include "oneflow/core/register/tensor_slice_view.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/graph/local_ring_boxing_task_node.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"

namespace oneflow {

SubTskGphBuilderStatus LocalRingSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, const ParallelDesc& src_parallel_desc,
    const ParallelDesc& dst_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& src_sbp_parallel,
    const SbpParallel& dst_sbp_parallel) const {
  if (!dst_parallel_desc.EqualsIgnoringPolicy(src_parallel_desc)) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  const ParallelDesc& parallel_desc = src_parallel_desc;
  if (parallel_desc.device_type() != DeviceType::kGPU) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  if (parallel_desc.parallel_num() <= 1) { return SubTskGphBuilderStatus::MakeStatusError(); }
  if (parallel_desc.sorted_machine_ids().size() > 1) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  const auto RingNextParallelId = [&parallel_desc](int64_t parallel_id) -> int64_t {
    return (parallel_id + 1) % parallel_desc.parallel_num();
  };
  std::vector<int64_t> ring;
  FOR_RANGE(int64_t, i, 0, parallel_desc.parallel_num()) {
    ring.push_back((i + 1) % parallel_desc.parallel_num());
  }
  const auto RingPrevParallelId = [&parallel_desc](int64_t parallel_id) -> int64_t {
    return (parallel_desc.parallel_num() + parallel_id - 1) % parallel_desc.parallel_num();
  };
  if (SubTskGphBuilderUtil::HasEmptySliceIfSplit(src_parallel_desc.parallel_num(), src_sbp_parallel,
                                                 logical_blob_desc)) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  if (SubTskGphBuilderUtil::HasEmptySliceIfSplit(dst_parallel_desc.parallel_num(), dst_sbp_parallel,
                                                 logical_blob_desc)) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  if (dst_parallel_desc.parallel_num() > logical_blob_desc.shape().elem_cnt()) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  std::vector<LocalRingBoxingTaskNode*> boxing_nodes;
  FOR_RANGE(int64_t, i, 0, parallel_desc.parallel_num()) {
    boxing_nodes.push_back(ctx->task_graph()->NewNode<LocalRingBoxingTaskNode>());
  }
  FOR_RANGE(int64_t, i, 0, parallel_desc.parallel_num()) {
    boxing_nodes.at(i)->Init(
        parallel_desc.sorted_machine_ids().at(0),
        Global<IDMgr>::Get()->GetGpuMixThrdId(sorted_src_comp_tasks.at(i)->GpuPhyId()), lbi,
        boxing_nodes.at(RingPrevParallelId(i)), ring);
  }
  ctx->NaiveConnectAll121(sorted_src_comp_tasks, boxing_nodes);
  ctx->NaiveConnectAll121(boxing_nodes, sorted_dst_comp_tasks);

  return SubTskGphBuilderStatus::MakeStatusOK();
}

}  // namespace oneflow
