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
    SubTskGphBuilderCtx* ctx, const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, const ParallelDesc& src_parallel_desc,
    const ParallelDesc& dst_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& src_sbp_parallel,
    const SbpParallel& dst_sbp_parallel) const {
  if ((src_parallel_desc.parallel_num() == 1 || src_sbp_parallel.has_broadcast_parallel())
      && dst_parallel_desc.parallel_num() != 1 && dst_sbp_parallel.has_partial_sum_parallel()) {
    HashMap<CompTaskNode*, CompTaskNode*> dst_node2nearest_src_node;
    int64_t nearest_dst_node_idx = -1;
    int64_t nearest_dst_node_distance = -1;
    std::vector<CompTaskNode*> nearest_src_comp_tasks;
    for (int64_t dst_node_idx = 0; dst_node_idx < sorted_dst_comp_tasks.size(); ++dst_node_idx) {
      CompTaskNode* dst_node = sorted_dst_comp_tasks.at(dst_node_idx);
      const int64_t nearest_src_node_idx =
          SubTskGphBuilderUtil::FindNearestNodeIndex(sorted_src_comp_tasks, dst_node);
      CHECK_NE_OR_RETURN(nearest_src_node_idx, -1);
      CompTaskNode* nearest_src_node = sorted_src_comp_tasks.at(nearest_src_node_idx);
      CHECK_OR_RETURN(dst_node2nearest_src_node.emplace(dst_node, nearest_src_node).second);
      const int64_t distance = SubTskGphBuilderUtil::GetDistance(nearest_src_node, dst_node);
      if (nearest_dst_node_idx == -1 || distance < nearest_dst_node_distance) {
        nearest_dst_node_idx = dst_node_idx;
        nearest_dst_node_distance = distance;
      }
    }
    for (int64_t dst_node_idx = 0; dst_node_idx < sorted_dst_comp_tasks.size(); ++dst_node_idx) {
      CompTaskNode* dst_node = sorted_dst_comp_tasks.at(dst_node_idx);
      CompTaskNode* nearest_src_node = dst_node2nearest_src_node.at(dst_node);
      if (dst_node_idx == nearest_dst_node_idx) {
        TaskNode* proxy = ctx->GetProxyNode(nearest_src_node, nearest_src_node->MemZoneId121(),
                                            dst_node->machine_id(), dst_node->MemZoneId121());
        Connect<TaskNode>(proxy, ctx->task_graph()->NewEdge(), dst_node);
      } else {
        auto* zeros_node = ctx->task_graph()->NewNode<BoxingZerosTaskNode>();
        zeros_node->Init(dst_node->machine_id(), dst_node->thrd_id(), dst_node->area_id(), lbi,
                         logical_blob_desc.shape(), logical_blob_desc.data_type(),
                         *nearest_src_node->logical_node()->out_blob_time_shape());
        nearest_src_node->BuildCtrlRegstDesc(zeros_node);
        Connect<TaskNode>(nearest_src_node, ctx->task_graph()->NewEdge(), zeros_node);
        Connect<TaskNode>(zeros_node, ctx->task_graph()->NewEdge(), dst_node);
      }
    }
    return TRY(BuildSubTskGphBuilderStatus(sorted_src_comp_tasks.front(),
                                           sorted_dst_comp_tasks.front(), src_parallel_desc,
                                           dst_parallel_desc, src_sbp_parallel, dst_sbp_parallel,
                                           lbi, logical_blob_desc, "NaiveB2PSubTskGphBuilder", ""));
  } else {
    return Error::BoxingNotSupportedError();
  }
}

}  // namespace oneflow
