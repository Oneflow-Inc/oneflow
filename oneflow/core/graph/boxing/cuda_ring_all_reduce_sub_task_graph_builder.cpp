#include "oneflow/core/graph/boxing/cuda_ring_all_reduce_sub_task_graph_builder.h"
#include "oneflow/core/graph/cuda_ring_all_reduce_task_node.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"
#include "oneflow/core/graph/cuda_copy_peer_task_node.h"

namespace oneflow {

SubTskGphBuilderStatus CudaRingAllReduceSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, const ParallelDesc& src_parallel_desc,
    const ParallelDesc& dst_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& src_sbp_parallel,
    const SbpParallel& dst_sbp_parallel) const {
  const int64_t num_rings = 2;
  if (SubTskGphBuilderUtil::BlobHasDynamicShape(logical_blob_desc)) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  if (!dst_parallel_desc.EqualsIgnoringPolicy(src_parallel_desc)) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  const ParallelDesc& parallel_desc = src_parallel_desc;
  if (parallel_desc.device_type() != DeviceType::kGPU) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  if (parallel_desc.sorted_machine_ids().size() > 1) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  if (parallel_desc.parallel_num() <= 1) { return SubTskGphBuilderStatus::MakeStatusError(); }
  if (!SubTskGphBuilderUtil::IsBoxingP2B(src_sbp_parallel, dst_sbp_parallel)) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  if (dst_parallel_desc.parallel_num() * num_rings > logical_blob_desc.shape().elem_cnt()) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  const auto RingNextParallelId = [&parallel_desc](int64_t parallel_id) -> int64_t {
    return (parallel_id + 1) % parallel_desc.parallel_num();
  };
  std::vector<int64_t> ring;
  FOR_RANGE(int64_t, i, 0, parallel_desc.parallel_num()) {
    ring.push_back((i + 1) % parallel_desc.parallel_num());
  }
  std::vector<CudaRingAllReduceTaskNode*> boxing_nodes;
  FOR_RANGE(int64_t, i, 0, parallel_desc.parallel_num()) {
    auto ring_all_reduce_task_node = ctx->task_graph()->NewNode<CudaRingAllReduceTaskNode>();
    ParallelContext parallel_ctx;
    parallel_ctx.set_parallel_num(parallel_desc.parallel_num());
    parallel_ctx.set_parallel_id(i);
    parallel_ctx.set_policy(ParallelPolicy::kDataParallel);
    ring_all_reduce_task_node->Init(
        parallel_desc.MachineIdForParallelId(i),
        Global<IDMgr>::Get()->GetGpuBoxingH2DThrdId(parallel_desc.DeviceIdForParallelId(i)), lbi,
        logical_blob_desc.shape(), parallel_ctx);
    boxing_nodes.push_back(ring_all_reduce_task_node);
  }

  FOR_RANGE(int64_t, ring_id, 0, num_rings) {
    std::vector<TaskNode*> send_to(parallel_desc.parallel_num());
    std::vector<TaskNode*> recv_from(parallel_desc.parallel_num());
    FOR_RANGE(int64_t, i, 0, parallel_desc.parallel_num()) {
      int64_t next_id = RingNextParallelId(i);
      send_to[i] = boxing_nodes.at(next_id);
      recv_from[next_id] = boxing_nodes.at(i);
    }
    FOR_RANGE(int64_t, i, 0, parallel_desc.parallel_num()) {
      boxing_nodes.at(i)->AddRing(ring, send_to[i], recv_from[i]);
    }
  }
  ctx->NaiveConnectAll121(sorted_src_comp_tasks, boxing_nodes);
  ctx->NaiveConnectAll121(boxing_nodes, sorted_dst_comp_tasks);
  return SubTskGphBuilderStatus::MakeStatusOK();
}

}  // namespace oneflow
