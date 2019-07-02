#include "oneflow/core/graph/boxing/multi_ring_all_reduce_sub_task_graph_builder.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/graph/multi_ring_all_reduce_task_node.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"

namespace oneflow {

SubTskGphBuilderStatus MultiRingAllReduceSubTskGphBuilder::Build(
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
  std::vector<MultiRingAllReduceTaskNode*> boxing_nodes;
  FOR_RANGE(int64_t, i, 0, parallel_desc.parallel_num()) {
    auto ring_all_reduce_task_node = ctx->task_graph()->NewNode<MultiRingAllReduceTaskNode>();
    ParallelContext parallel_ctx;
    parallel_ctx.set_parallel_num(parallel_desc.parallel_num());
    parallel_ctx.set_parallel_id(i);
    parallel_ctx.set_policy(ParallelPolicy::kDataParallel);
    ring_all_reduce_task_node->Init(parallel_desc.MachineIdForParallelId(i),
                                    parallel_desc.DeviceIdForParallelId(i), lbi,
                                    logical_blob_desc.shape(), parallel_ctx);
    boxing_nodes.push_back(ring_all_reduce_task_node);
  }

  FOR_RANGE(int64_t, ring_id, 0, num_rings) {
    std::vector<TaskNode*> send_to(parallel_desc.parallel_num());
    std::vector<TaskNode*> recv_from(parallel_desc.parallel_num());
    FOR_RANGE(int64_t, i, 0, parallel_desc.parallel_num()) {
      int64_t next_id = RingNextParallelId(i);
      if (parallel_desc.MachineIdForParallelId(i)
          == parallel_desc.MachineIdForParallelId(next_id)) {
        CopyHdTaskNode* copy_d2d_task_node = ctx->task_graph()->NewNode<CopyHdTaskNode>();
        copy_d2d_task_node->Init(CopyHdOpConf::D2D, parallel_desc.MachineIdForParallelId(next_id),
                                 parallel_desc.DeviceIdForParallelId(next_id));
        Connect<TaskNode>(boxing_nodes.at(i), ctx->task_graph()->NewEdge(), copy_d2d_task_node);
        send_to[i] = copy_d2d_task_node;
        recv_from[next_id] = copy_d2d_task_node;
      } else {
        CopyHdTaskNode* copy_d2h_task_node = ctx->task_graph()->NewNode<CopyHdTaskNode>();
        copy_d2h_task_node->Init(CopyHdOpConf::D2H, parallel_desc.MachineIdForParallelId(i),
                                 parallel_desc.DeviceIdForParallelId(i));
        Connect<TaskNode>(boxing_nodes.at(i), ctx->task_graph()->NewEdge(), copy_d2h_task_node);
        CopyCommNetTaskNode* copy_comm_net_task_node =
            ctx->task_graph()->NewNode<CopyCommNetTaskNode>();
        copy_comm_net_task_node->Init(parallel_desc.MachineIdForParallelId(next_id),
                                      parallel_desc.MachineIdForParallelId(i));
        Connect<TaskNode>(copy_d2h_task_node, ctx->task_graph()->NewEdge(),
                          copy_comm_net_task_node);
        CopyHdTaskNode* copy_h2d_task_node = ctx->task_graph()->NewNode<CopyHdTaskNode>();
        copy_h2d_task_node->Init(CopyHdOpConf::H2D, parallel_desc.MachineIdForParallelId(next_id),
                                 parallel_desc.DeviceIdForParallelId(next_id));
        Connect<TaskNode>(copy_comm_net_task_node, ctx->task_graph()->NewEdge(),
                          copy_h2d_task_node);
        send_to[i] = copy_d2h_task_node;
        recv_from[next_id] = copy_h2d_task_node;
      }
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
