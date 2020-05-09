#include "oneflow/core/graph/boxing/naive_b2b_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"

namespace oneflow {

namespace {
constexpr int64_t kDistanceSameDevice = 0;
constexpr int64_t kDistanceSameMachine = 1;
constexpr int64_t kDistanceDiffMachine = 2;
constexpr int64_t kDistanceMax = 3;

int64_t GetDistance(const CompTaskNode* src, const CompTaskNode* dst) {
  if (src->machine_id() != dst->machine_id()) {
    return kDistanceDiffMachine;
  } else if (src->device_type() != dst->device_type()) {
    return kDistanceSameMachine;
  } else if (src->device_type() == DeviceType::kCPU) {
    return kDistanceSameDevice;
  } else {
    if (Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(src->thrd_id())
        == Global<IDMgr>::Get()->GetGpuPhyIdFromThrdId(dst->thrd_id())) {
      return kDistanceSameDevice;
    } else {
      return kDistanceSameMachine;
    }
  }
}

}  // namespace

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
      CompTaskNode* nearest_src_node = nullptr;
      int64_t nearest_distance = kDistanceMax;
      for (CompTaskNode* src_node : sorted_src_comp_tasks) {
        int64_t distance = GetDistance(src_node, dst_node);
        if (distance < nearest_distance) {
          nearest_src_node = src_node;
          nearest_distance = distance;
        }
      }
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
