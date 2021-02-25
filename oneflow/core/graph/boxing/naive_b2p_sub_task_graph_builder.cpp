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
#include "oneflow/core/common/id_util.h"
#include "oneflow/core/device/stream_index.h"

namespace oneflow {

Maybe<SubTskGphBuilderStatus> NaiveB2PSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
    std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
    const SbpParallel& out_sbp_parallel, const Shape& time_shape) const {
  if ((in_parallel_desc.parallel_num() == 1 || in_sbp_parallel.has_broadcast_parallel())
      && out_parallel_desc.parallel_num() != 1 && out_sbp_parallel.has_partial_sum_parallel()) {
    HashMap<int64_t, int64_t> out_id2nearest_in_id;
    int64_t nearest_out_node_idx = -1;
    int64_t nearest_out_node_distance = -1;

    FOR_RANGE(int64_t, out_id, 0, out_parallel_desc.parallel_num()) {
      const int64_t nearest_in_parallel_id = SubTskGphBuilderUtil::FindNearestSrcParallelId(
          in_parallel_desc, out_parallel_desc, out_id);
      out_id2nearest_in_id.emplace(out_id, nearest_in_parallel_id);
      const int64_t distance = SubTskGphBuilderUtil::GetDistance(
          in_parallel_desc, nearest_in_parallel_id, out_parallel_desc, out_id);
      if (nearest_out_node_idx == -1 || distance < nearest_out_node_distance) {
        nearest_out_node_idx = out_id;
        nearest_out_node_distance = distance;
      }
    }
    FOR_RANGE(int64_t, out_id, 0, out_parallel_desc.parallel_num()) {
      const int64_t nearest_in_id = out_id2nearest_in_id.at(out_id);
      TaskNode* nearest_in_node = sorted_in_tasks.at(nearest_in_id);
      if (out_id == nearest_out_node_idx) {
        TaskNode* proxy = ctx->GetProxyNode(nearest_in_node, nearest_in_node->MemZoneId121(),
                                            out_parallel_desc, out_id);

        sorted_out_tasks->push_back(proxy);
      } else {
        const int64_t out_machine_id = CHECK_JUST(out_parallel_desc.MachineId4ParallelId(out_id));
        const int64_t out_dev_phy_id = CHECK_JUST(out_parallel_desc.DeviceId4ParallelId(out_id));
        int64_t thrd_id;
        if (out_parallel_desc.device_type() == DeviceType::kGPU) {
#ifdef WITH_CUDA
          ProcessId process_id{static_cast<uint32_t>(out_machine_id), 0};
          DeviceId device_id{DeviceType::kGPU, static_cast<uint32_t>(out_dev_phy_id)};
          auto* stream_index_generator =
              Global<IDMgr>::Get()->GetStreamIndexGeneratorManager()->GetGenerator(process_id,
                                                                                   device_id);
          CHECK_NOTNULL(stream_index_generator);
          uint32_t stream_index = stream_index_generator->GenerateComputeStreamIndex();
          thrd_id = SerializeStreamIdToInt64(StreamId{device_id, stream_index});
#else
          UNIMPLEMENTED();
#endif
        } else if (out_parallel_desc.device_type() == DeviceType::kCPU) {
          thrd_id = Global<IDMgr>::Get()->PickCpuThrdIdEvenly(out_machine_id);
        } else {
          UNIMPLEMENTED();
        }
        auto* zeros_node = ctx->task_graph()->NewNode<BoxingZerosTaskNode>();
        zeros_node->Init(out_machine_id, thrd_id, NewAreaId(), lbi, logical_blob_desc.shape(),
                         logical_blob_desc.data_type(), time_shape);
        nearest_in_node->BuildCtrlRegstDesc(zeros_node);
        Connect<TaskNode>(nearest_in_node, ctx->task_graph()->NewEdge(), zeros_node);
        sorted_out_tasks->push_back(zeros_node);
      }
    }
    return TRY(BuildSubTskGphBuilderStatus("NaiveB2PSubTskGphBuilder", ""));
  } else {
    return Error::BoxingNotSupportedError();
  }
}

}  // namespace oneflow
