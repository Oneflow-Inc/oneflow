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
    SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
    std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
    const SbpParallel& out_sbp_parallel, const Shape& time_shape) const {
  if ((in_parallel_desc.parallel_num() == 1 && out_parallel_desc.parallel_num() == 1)
      || (in_parallel_desc.parallel_num() == out_parallel_desc.parallel_num()
          && in_sbp_parallel == out_sbp_parallel)) {
    for (int64_t i = 0; i < in_parallel_desc.parallel_num(); ++i) {
      TaskNode* in_node = sorted_in_tasks.at(i);
      TaskNode* proxy = ctx->task_graph()->GetProxyNode(in_node, lbi, out_parallel_desc, i);
      sorted_out_tasks->push_back(proxy);
    }
    return TRY(BuildSubTskGphBuilderStatus("OneToOneSubTskGphBuilder", ""));
  } else {
    return Error::BoxingNotSupportedError();
  }
}

}  // namespace oneflow
