#include "oneflow/core/graph/boxing/local_peer_sub_task_graph_builder.h"
#include "oneflow/core/register/tensor_slice_view.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/graph/slice_boxing_task_node.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"

namespace oneflow {

SubTskGphBuilderStatus LocalPeerSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, const ParallelDesc& src_parallel_desc,
    const ParallelDesc& dst_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& src_sbp_parallel,
    const SbpParallel& dst_sbp_parallel) const {
  if (src_parallel_desc.sorted_machine_ids().size() != 1
      || dst_parallel_desc.sorted_machine_ids().size() != 1) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  if (src_parallel_desc.sorted_machine_ids().at(0)
      != dst_parallel_desc.sorted_machine_ids().at(0)) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  if (!SubTskGphBuilderUtil::IsDeviceTypeCPUOrGPU(src_parallel_desc)) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  if (!SubTskGphBuilderUtil::IsDeviceTypeCPUOrGPU(dst_parallel_desc)) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  std::vector<TaskNode*> from_nodes;
  ParallelDesc from_parallel_desc = src_parallel_desc;
  if (src_parallel_desc.device_type() == DeviceType::kGPU
      && dst_parallel_desc.device_type() == DeviceType::kCPU) {
    FOR_RANGE(int64_t, i, 0, src_parallel_desc.parallel_num()) {
      TaskNode* src_node = sorted_src_comp_tasks.at(i);
      CopyHdTaskNode* copy_task = ctx->task_graph()->NewNode<CopyHdTaskNode>();
      copy_task->Init(CopyHdOpConf::D2H, src_node->machine_id(), src_node->GpuPhyId());
      Connect<TaskNode>(src_node, ctx->task_graph()->NewEdge(), copy_task);
      from_nodes.push_back(copy_task);
    }
    from_parallel_desc = SubTskGphBuilderUtil::CloneParallelDescWithNewDeviceType(src_parallel_desc,
                                                                                  DeviceType::kCPU);
  } else {
    FOR_RANGE(int64_t, i, 0, src_parallel_desc.parallel_num()) {
      TaskNode* src_node = sorted_src_comp_tasks.at(i);
      from_nodes.push_back(src_node);
    }
  }

  std::vector<TaskNode*> boxed_nodes;
  const auto BuildSubTaskGph = [&ctx, &lbi, &logical_blob_desc](
                                   const ParallelDesc& src_pd, const ParallelDesc& dst_pd,
                                   const SbpParallel& src_sbp, const SbpParallel& dst_sbp,
                                   const std::vector<TaskNode*> in_nodes,
                                   std::vector<TaskNode*>* out_nodes) {
    const std::vector<TensorSliceView> src_views =
        SubTskGphBuilderUtil::GetTensorSliceView(src_pd.parallel_num(), src_sbp, logical_blob_desc);
    const std::vector<TensorSliceView> dst_views =
        SubTskGphBuilderUtil::GetTensorSliceView(dst_pd.parallel_num(), dst_sbp, logical_blob_desc);
    const SliceBoxingTaskMode mode = src_sbp.has_partial_sum_parallel()
                                         ? SliceBoxingTaskMode::kSliceBoxingTaskModeAdd
                                         : SliceBoxingTaskMode::kSliceBoxingTaskModeCopy;
    FOR_RANGE(int64_t, i, 0, dst_pd.parallel_num()) {
      SliceBoxingTaskNode* copy_task_node = ctx->task_graph()->NewNode<SliceBoxingTaskNode>();
      const int64_t machine_id = dst_pd.MachineIdForParallelId(i);
      int64_t thrd_id = -1;
      if (dst_pd.device_type() == DeviceType::kCPU) {
        thrd_id = Global<IDMgr>::Get()->PickCpuThrdIdEvenly(machine_id);
      } else if (dst_pd.device_type() == DeviceType::kGPU) {
        thrd_id = Global<IDMgr>::Get()->GetGpuMixThrdId(dst_pd.DeviceIdForParallelId(i));
      } else {
        UNIMPLEMENTED();
      }
      copy_task_node->Init(lbi, dst_views.at(i), mode, machine_id, thrd_id);
      FOR_RANGE(int64_t, src_id, 0, src_pd.parallel_num()) {
        copy_task_node->ConnectToSrcNodeWithSlice(in_nodes.at(src_id), ctx->task_graph()->NewEdge(),
                                                  src_views.at(src_id));
      }
      out_nodes->push_back(copy_task_node);
    }
  };
  if (src_sbp_parallel.has_partial_sum_parallel() && dst_sbp_parallel.has_broadcast_parallel()) {
    std::vector<TaskNode*> split_nodes;
    SbpParallel split_sbp;
    split_sbp.mutable_split_parallel()->set_axis(0);
    BuildSubTaskGph(from_parallel_desc, dst_parallel_desc, src_sbp_parallel, split_sbp, from_nodes,
                    &split_nodes);
    BuildSubTaskGph(dst_parallel_desc, dst_parallel_desc, split_sbp, dst_sbp_parallel, split_nodes,
                    &boxed_nodes);
  } else {
    BuildSubTaskGph(from_parallel_desc, dst_parallel_desc, src_sbp_parallel, dst_sbp_parallel,
                    from_nodes, &boxed_nodes);
  }
  CHECK_EQ(boxed_nodes.size(), sorted_dst_comp_tasks.size());
  FOR_RANGE(int64_t, i, 0, sorted_dst_comp_tasks.size()) {
    Connect<TaskNode>(boxed_nodes.at(i), ctx->task_graph()->NewEdge(), sorted_dst_comp_tasks.at(i));
  }
  return SubTskGphBuilderStatus::MakeStatusOK();
}

}  // namespace oneflow
