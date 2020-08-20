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
    SubTskGphBuilderCtx* ctx, const std::vector<CompTaskNode*>& sorted_src_comp_tasks,
    const std::vector<CompTaskNode*>& sorted_dst_comp_tasks, const ParallelDesc& src_parallel_desc,
    const ParallelDesc& dst_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& src_sbp_parallel,
    const SbpParallel& dst_sbp_parallel) const {
  for (const auto& builder : builders_) {
    Maybe<SubTskGphBuilderStatus> boxing_builder_status = TRY(builder->Build(
        ctx, sorted_src_comp_tasks, sorted_dst_comp_tasks, src_parallel_desc, dst_parallel_desc,
        lbi, logical_blob_desc, src_sbp_parallel, dst_sbp_parallel));
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
