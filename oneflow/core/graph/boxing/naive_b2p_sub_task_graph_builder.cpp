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
#include "oneflow/core/graph/boxing/naive_b2p_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"
#include "oneflow/core/graph/boxing_zeros_task_node.h"

namespace oneflow {

Maybe<SubTskGphBuilderStatus> NaiveB2PSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
    std::vector<TaskNode*>* sorted_dst_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_dst_ctrl_in_tasks,
    const ParallelDesc& src_parallel_desc, const ParallelDesc& dst_parallel_desc,
    const LogicalBlobId& lbi, const BlobDesc& logical_blob_desc,
    const SbpParallel& src_sbp_parallel, const SbpParallel& dst_sbp_parallel,
    const Shape& time_shape) const {
  if ((src_parallel_desc.parallel_num() == 1 || src_sbp_parallel.has_broadcast_parallel())
      && dst_parallel_desc.parallel_num() != 1 && dst_sbp_parallel.has_partial_sum_parallel()) {
    HashMap<int64_t, int64_t> dst_id2nearest_src_id;
    int64_t nearest_dst_node_idx = -1;
    int64_t nearest_dst_node_distance = -1;

    FOR_RANGE(int64_t, out_id, 0, dst_parallel_desc.parallel_num()) {
      const int64_t nearest_src_parallel_id =
          SubTskGphBuilderUtil::FindNearestParallelId(src_parallel_desc, dst_parallel_desc, out_id);
      dst_id2nearest_src_id.emplace(out_id, nearest_src_parallel_id);
      const int64_t distance = SubTskGphBuilderUtil::GetDistance(
          src_parallel_desc, nearest_src_parallel_id, dst_parallel_desc, out_id);
      if (nearest_dst_node_idx == -1 || distance < nearest_dst_node_distance) {
        nearest_dst_node_idx = out_id;
        nearest_dst_node_distance = distance;
      }
    }
    FOR_RANGE(int64_t, out_id, 0, dst_parallel_desc.parallel_num()) {
      const int64_t nearest_src_id = dst_id2nearest_src_id.at(out_id);
      TaskNode* nearest_in_node = sorted_in_tasks.at(nearest_src_id);
      if (out_id == nearest_dst_node_idx) {
        TaskNode* proxy = ctx->GetProxyNode(nearest_in_node, nearest_in_node->MemZoneId121(),
                                            dst_parallel_desc, out_id);

        sorted_dst_tasks->push_back(proxy);
      } else {
        const int64_t dst_machine_id = CHECK_JUST(dst_parallel_desc.MachineId4ParallelId(out_id));
        const int64_t dst_dev_phy_id = CHECK_JUST(dst_parallel_desc.DeviceId4ParallelId(out_id));
        int64_t thrd_id;
        if (dst_parallel_desc.device_type() == DeviceType::kGPU) {
#ifdef WITH_CUDA
          thrd_id = Global<IDMgr>::Get()->GetGpuComputeThrdId(dst_dev_phy_id);
#else
          UNIMPLEMENTED();
#endif
        } else {
          std::vector<int64_t> cpu_device_offset(
              Global<ResourceDesc, ForSession>::Get()->TotalMachineNum(), 0);
          int64_t& offset = cpu_device_offset.at(dst_machine_id);
          thrd_id = Global<IDMgr>::Get()->GetCpuDeviceThrdId(offset);
        }
        auto* zeros_node = ctx->task_graph()->NewNode<BoxingZerosTaskNode>();
        zeros_node->Init(dst_machine_id, thrd_id, NewAreaId(), lbi, logical_blob_desc.shape(),
                         logical_blob_desc.data_type(), time_shape);
        nearest_in_node->BuildCtrlRegstDesc(zeros_node);
        Connect<TaskNode>(nearest_in_node, ctx->task_graph()->NewEdge(), zeros_node);
        sorted_dst_tasks->push_back(zeros_node);
      }
    }
    return TRY(BuildSubTskGphBuilderStatus("NaiveB2PSubTskGphBuilder", ""));
  } else {
    return Error::BoxingNotSupportedError();
  }
}

}  // namespace oneflow
