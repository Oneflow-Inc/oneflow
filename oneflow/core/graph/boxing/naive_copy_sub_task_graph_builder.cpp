#include "oneflow/core/graph/boxing/naive_copy_sub_task_graph_builder.h"
#include "oneflow/core/register/tensor_slice_view.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/graph/slice_boxing_task_node.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"

namespace oneflow {

SubTskGphBuilderStatus NaiveCopySubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, const ParallelDesc& src_parallel_desc,
    const ParallelDesc& dst_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& src_sbp_parallel,
    const SbpParallel& dst_sbp_parallel) const {
  if (!SubTskGphBuilderUtil::IsDeviceTypeCPUOrGPU(src_parallel_desc)) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  if (!SubTskGphBuilderUtil::IsDeviceTypeCPUOrGPU(dst_parallel_desc)) {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
  const auto GetBoxingGpuThrdId = [](const int64_t dev_id, CudaWorkType work_type) -> int64_t {
    if (work_type == CudaWorkType::kBoxingH2D) {
      return Global<IDMgr>::Get()->GetGpuBoxingH2DThrdId(dev_id);
    } else if (work_type == CudaWorkType::kBoxingD2H) {
      return Global<IDMgr>::Get()->GetGpuBoxingD2HThrdId(dev_id);
    } else {
      return Global<IDMgr>::Get()->GetGpuMixThrdId(dev_id);
    }
  };
  const auto NewEdge = [&ctx]() -> TaskEdge* { return ctx->task_graph()->NewEdge(); };
  const auto CreateBoxingNode121 = [&ctx, &lbi, &GetBoxingGpuThrdId](
                                       const ParallelDesc& pd, const int64_t parallel_id,
                                       const TensorSliceView& slice,
                                       SliceBoxingTaskMode mode) -> SliceBoxingTaskNode* {
    SliceBoxingTaskNode* node = ctx->task_graph()->NewNode<SliceBoxingTaskNode>();
    const int64_t machine_id = pd.MachineIdForParallelId(parallel_id);
    int64_t thrd_id = -1;
    if (pd.device_type() == DeviceType::kCPU) {
      thrd_id = Global<IDMgr>::Get()->PickCpuThrdIdEvenly(machine_id);
    } else if (pd.device_type() == DeviceType::kGPU) {
      thrd_id = GetBoxingGpuThrdId(pd.DeviceIdForParallelId(parallel_id), CudaWorkType::kBoxingH2D);
    } else {
      UNIMPLEMENTED();
    }
    node->Init(lbi, slice, mode, machine_id, thrd_id);
    return node;
  };
  if (SubTskGphBuilderUtil::IsBoxingP2B(src_sbp_parallel, dst_sbp_parallel)) {
    const TensorSliceView slice =
        SubTskGphBuilderUtil::GetBroadcastTensorSliceView(logical_blob_desc);
    FOR_RANGE(int64_t, out_id, 0, dst_parallel_desc.parallel_num()) {
      SliceBoxingTaskNode* out_node =
          CreateBoxingNode121(dst_parallel_desc, out_id, slice, kSliceBoxingTaskModeAdd);
      FOR_RANGE(int64_t, in_id, 0, src_parallel_desc.parallel_num()) {
        TaskNode* in_node = sorted_src_comp_tasks.at(in_id);
        TaskNode* proxy = ctx->GetProxyNode(in_node, GetDefaultMemCase(in_node),
                                            out_node->machine_id(), GetDefaultMemCase(out_node));
        out_node->ConnectToSrcNodeWithSlice(proxy, NewEdge(), slice);
      }
      Connect<TaskNode>(out_node, NewEdge(), sorted_dst_comp_tasks.at(out_id));
    }
    return SubTskGphBuilderStatus::MakeStatusOK();
  } else {
    return SubTskGphBuilderStatus::MakeStatusError();
  }
}

}  // namespace oneflow
