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
#include "oneflow/core/graph/boxing/chain_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/sub_task_graph_builder_util.h"

namespace oneflow {

Maybe<SubTskGphBuilderStatus> ChainSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
    std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
    const SbpParallel& out_sbp_parallel, const Shape& time_shape) const {
  for (const auto& builder : builders_) {
    Maybe<SubTskGphBuilderStatus> boxing_builder_status = TRY(builder->Build(
        ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks, in_parallel_desc,
        out_parallel_desc, lbi, logical_blob_desc, in_sbp_parallel, out_sbp_parallel, time_shape));
    if (!boxing_builder_status.IsOk()
        && SubTskGphBuilderUtil::IsErrorBoxingNotSupported(*boxing_builder_status.error())) {
      continue;
    } else {
      return boxing_builder_status;
    }
  }
  return Error::BoxingNotSupportedError();
}

}  // namespace oneflow
