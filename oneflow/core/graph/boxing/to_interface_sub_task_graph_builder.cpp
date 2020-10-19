/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/graph/boxing/to_interface_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"
#include "oneflow/core/graph/slice_boxing_task_node.h"

namespace oneflow {

Maybe<SubTskGphBuilderStatus> ToInterfaceSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, const ParallelDesc& src_parallel_desc,
    const ParallelDesc& dst_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& src_sbp_parallel,
    const SbpParallel& dst_sbp_parallel) const {
  const LogicalNode* dst_logical_node = sorted_dst_comp_tasks.front()->logical_node();
  if (dst_logical_node->op_vec().size() != 1) { return Error::BoxingNotSupportedError(); }
  if (!IsClassRegistered<int32_t, IsInterfaceOpConf4OpTypeCase>(
          dst_logical_node->SoleOp()->op_conf().op_type_case())) {
    return Error::BoxingNotSupportedError();
  }
  if ((src_parallel_desc.parallel_num() == 1 || src_sbp_parallel.has_broadcast_parallel())
      && (dst_parallel_desc.parallel_num() == 1 || dst_sbp_parallel.has_broadcast_parallel())) {
    std::vector<CompTaskNode*> nearest_src_comp_tasks;
    for (CompTaskNode* dst_node : sorted_dst_comp_tasks) {
      CompTaskNode* nearest_src_node =
          SubTskGphBuilderUtil::FindNearestNode(sorted_src_comp_tasks, dst_node);
      CHECK_NOTNULL(nearest_src_node);
      if (SubTskGphBuilderUtil::IsOnSameGPU(nearest_src_node, dst_node)) {
        Connect<TaskNode>(nearest_src_node, ctx->task_graph()->NewEdge(), dst_node);
      } else {
        TaskNode* proxy =
            ctx->GetProxyNode(nearest_src_node, nearest_src_node->MemZoneId121(),
                              dst_node->machine_id(), Global<IDMgr>::Get()->CpuMemZoneId());
        Connect<TaskNode>(proxy, ctx->task_graph()->NewEdge(), dst_node);
      }
    }
    return TRY(BuildSubTskGphBuilderStatus(
        sorted_src_comp_tasks.front(), sorted_dst_comp_tasks.front(), src_parallel_desc,
        dst_parallel_desc, src_sbp_parallel, dst_sbp_parallel, lbi, logical_blob_desc,
        "ToInterfaceSubTskGphBuilder", "BuildSubTaskGphB2B"));
  } else if ((src_parallel_desc.parallel_num() == 1 || src_sbp_parallel.has_broadcast_parallel())
             && (dst_parallel_desc.parallel_num() > 1 || dst_sbp_parallel.has_split_parallel())) {
    const TensorSliceView in_slice =
        SubTskGphBuilderUtil::GetBroadcastTensorSliceView(logical_blob_desc);
    const std::vector<TensorSliceView> out_slices = SubTskGphBuilderUtil::GetTensorSliceView(
        dst_parallel_desc.parallel_num(), dst_sbp_parallel, logical_blob_desc);
    FOR_RANGE(int64_t, out_id, 0, dst_parallel_desc.parallel_num()) {
      const TensorSliceView& out_slice = out_slices.at(out_id);
      CompTaskNode* dst_node = sorted_dst_comp_tasks.at(out_id);
      const int64_t nearest_idx =
          SubTskGphBuilderUtil::FindNearestNodeIndex(sorted_src_comp_tasks, dst_node);
      CompTaskNode* src_node = sorted_src_comp_tasks.at(nearest_idx);
      SliceBoxingTaskNode* slice_node = ctx->task_graph()->NewNode<SliceBoxingTaskNode>();
      const auto src_machine_id = CHECK_JUST(src_parallel_desc.MachineId4ParallelId(0));
      if (src_parallel_desc.device_type() == DeviceType::kCPU) {
        slice_node->Init(lbi, out_slice, kSliceBoxingTaskModeCopy, src_machine_id,
                         Global<IDMgr>::Get()->PickCpuThrdIdEvenly(src_machine_id));
      } else if (src_parallel_desc.device_type() == DeviceType::kGPU) {
        slice_node->Init(lbi, out_slice, kSliceBoxingTaskModeCopy, src_machine_id,
                         Global<IDMgr>::Get()->GetGpuD2HThrdId(src_node->GpuPhyId()),
                         Global<IDMgr>::Get()->CpuMemZoneId());
      } else {
        UNIMPLEMENTED();
      }
      slice_node->ConnectToSrcNodeWithSlice(src_node, ctx->task_graph()->NewEdge(), in_slice);
      TaskNode* proxy =
          ctx->GetProxyNode(slice_node, slice_node->MemZoneId121(), dst_node->machine_id(),
                            Global<IDMgr>::Get()->CpuMemZoneId());
      Connect<TaskNode>(proxy, ctx->task_graph()->NewEdge(), dst_node);
    }
    return TRY(BuildSubTskGphBuilderStatus(
        sorted_src_comp_tasks.front(), sorted_dst_comp_tasks.front(), src_parallel_desc,
        dst_parallel_desc, src_sbp_parallel, dst_sbp_parallel, lbi, logical_blob_desc,
        "ToInterfaceSubTskGphBuilder", "BuildSubTaskGphB2S"));
  } else {
    return Error::BoxingNotSupportedError();
  }
}

}  // namespace oneflow
