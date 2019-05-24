#include "oneflow/core/graph/boxing/local_peer_boxing_builder.h"
#include "oneflow/core/register/tensor_partial_view.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/graph/boxing_v2_task_node.h"

namespace oneflow {

namespace {

bool IsDeviceTypeCPUOrGPU(const ParallelDesc& parallel_desc) {
  return parallel_desc.device_type() == DeviceType::kCPU
         || parallel_desc.device_type() == DeviceType::kGPU;
}

std::vector<TensorPartialView> GetTensorPartialView(const int64_t parallel_num,
                                                    const SbpParallel& sbp_parallel,
                                                    const BlobDesc& blob_desc) {
  std::vector<Range> ranges(blob_desc.shape().NumAxes());
  FOR_RANGE(int64_t, i, 0, blob_desc.shape().NumAxes()) {
    ranges[i].mut_begin() = 0;
    ranges[i].mut_end() = blob_desc.shape().At(i);
  }
  std::vector<TensorPartialView> views;
  if (sbp_parallel.has_partial_sum_parallel() || sbp_parallel.has_broadcast_parallel()) {
    FOR_RANGE(int64_t, i, 0, parallel_num) { views.emplace_back(ranges); }
  } else if (sbp_parallel.has_split_parallel()) {
    const int64_t axis = sbp_parallel.split_parallel().axis();
    const BalancedSplitter bs(blob_desc.shape().At(axis), parallel_num);
    FOR_RANGE(int64_t, i, 0, parallel_num) {
      ranges[axis] = bs.At(i);
      views.emplace_back(ranges);
    }
  } else {
    UNIMPLEMENTED();
  }
  return views;
}

}  // namespace

BoxingBuilderStatus LocalPeerBoxingBuilder::Build(
    BoxingBuilderCtx* ctx, const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, const ParallelDesc& src_parallel_desc,
    const ParallelDesc& dst_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& src_sbp_parallel,
    const SbpParallel& dst_sbp_parallel) const {
  if (src_parallel_desc.sorted_machine_ids().size() != 1
      || dst_parallel_desc.sorted_machine_ids().size() != 1) {
    return BoxingBuilderStatus::MakeStatusError();
  }
  if (src_parallel_desc.sorted_machine_ids().at(0)
      != dst_parallel_desc.sorted_machine_ids().at(0)) {
    return BoxingBuilderStatus::MakeStatusError();
  }
  if (!IsDeviceTypeCPUOrGPU(src_parallel_desc)) { return BoxingBuilderStatus::MakeStatusError(); }
  if (!IsDeviceTypeCPUOrGPU(dst_parallel_desc)) { return BoxingBuilderStatus::MakeStatusError(); }

  std::vector<TaskNode*> from_nodes;
  if (src_parallel_desc.device_type() == DeviceType::kGPU
      && dst_parallel_desc.device_type() == DeviceType::kCPU) {
    FOR_RANGE(int64_t, i, 0, src_parallel_desc.parallel_num()) {
      TaskNode* src_node = sorted_src_comp_tasks.at(i);
      CopyHdTaskNode* copy_task = ctx->task_graph()->NewNode<CopyHdTaskNode>();
      copy_task->Init(CopyHdOpConf::H2D, src_node->machine_id(), src_node->GpuPhyId());
      Connect<TaskNode>(src_node, ctx->task_graph()->NewEdge(), copy_task);
      from_nodes.push_back(copy_task);
    }
  } else {
    FOR_RANGE(int64_t, i, 0, src_parallel_desc.parallel_num()) {
      TaskNode* src_node = sorted_src_comp_tasks.at(i);
      from_nodes.push_back(src_node);
    }
  }

  const auto GetNewBoxingNodeThrdId = [&](const TaskNode* dst_node) -> int64_t {
    if (dst_parallel_desc.device_type() == kCPU) {
      return ctx->AllocateCpuThrdId(dst_node);
    } else if (dst_parallel_desc.device_type() == kGPU) {
      return Global<IDMgr>::Get()->GetGpuMixThrdId(dst_node->GpuPhyId());
    } else {
      UNIMPLEMENTED();
    }
  };
  const std::vector<TensorPartialView> src_views =
      GetTensorPartialView(src_parallel_desc.parallel_num(), src_sbp_parallel, logical_blob_desc);
  const std::vector<TensorPartialView> dst_views =
      GetTensorPartialView(dst_parallel_desc.parallel_num(), dst_sbp_parallel, logical_blob_desc);
  const BoxingV2TaskMode mode = src_sbp_parallel.has_partial_sum_parallel()
                                    ? BoxingV2TaskMode::kBoxingV2TaskModeAdd
                                    : kBoxingV2TaskModeCopy;
  FOR_RANGE(int64_t, i, 0, dst_parallel_desc.parallel_num()) {
    CompTaskNode* dst_comp_task_node = sorted_dst_comp_tasks.at(i);
    BoxingV2TaskNode* copy_task_node = ctx->task_graph()->NewNode<BoxingV2TaskNode>();
    copy_task_node->Init(lbi, dst_views.at(i), mode);
    copy_task_node->set_machine_id(dst_comp_task_node->machine_id());
    copy_task_node->set_thrd_id(GetNewBoxingNodeThrdId(dst_comp_task_node));
    copy_task_node->set_area_id(kBoundaryArea);
    Connect<TaskNode>(copy_task_node, ctx->task_graph()->NewEdge(), dst_comp_task_node);
    FOR_RANGE(int64_t, src_id, 0, src_parallel_desc.parallel_num()) {
      TaskEdge* edge = ctx->task_graph()->NewEdge();
      Connect<TaskNode>(from_nodes.at(src_id), edge, copy_task_node);
      copy_task_node->SetInDataEdgeView(edge, src_views.at(src_id));
    }
  }
  return BoxingBuilderStatus::MakeStatusOK();
}

}  // namespace oneflow
