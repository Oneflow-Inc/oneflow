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
#include "oneflow/core/graph/boxing/one_to_one_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"

namespace oneflow {

Maybe<SubTskGphBuilderStatus> OneToOneSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_src_tasks,
    std::vector<TaskNode*>* sorted_dst_tasks, const ParallelDesc& src_parallel_desc,
    const ParallelDesc& dst_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& src_sbp_parallel,
    const SbpParallel& dst_sbp_parallel, const Shape& time_shape) const {
  if ((src_parallel_desc.parallel_num() == 1 && dst_parallel_desc.parallel_num() == 1)
      || (src_parallel_desc.parallel_num() == dst_parallel_desc.parallel_num()
          && src_sbp_parallel == dst_sbp_parallel)) {
    for (int64_t i = 0; i < src_parallel_desc.parallel_num(); ++i) {
      const int64_t dst_machine_id = CHECK_JUST(dst_parallel_desc.MachineId4ParallelId(i));
      const int64_t dst_dev_phy_id = CHECK_JUST(dst_parallel_desc.DeviceId4ParallelId(i));
      const int64_t dst_mem_zone_id = SubTskGphBuilderUtil::GetMemZoneId(
          dst_machine_id, dst_dev_phy_id, dst_parallel_desc.device_type());
      TaskNode* src_node = sorted_src_tasks.at(i);
      // TODO(liujuncheng): use lbi
      TaskNode* proxy =
          ctx->GetProxyNode(src_node, src_node->MemZoneId121(), dst_machine_id, dst_mem_zone_id);
      sorted_dst_tasks->push_back(proxy);
    }
    return TRY(BuildSubTskGphBuilderStatus(sorted_src_tasks.front(), sorted_dst_tasks->front(),
                                           src_parallel_desc, dst_parallel_desc, src_sbp_parallel,
                                           dst_sbp_parallel, lbi, logical_blob_desc,
                                           "OneToOneSubTskGphBuilder", ""));
  } else {
    return Error::BoxingNotSupportedError();
  }
}

}  // namespace oneflow
