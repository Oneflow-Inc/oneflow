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
#include "oneflow/cambricon/graph/boxing/cncl_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/chain_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/ccl_sub_task_graph_builder.h"

namespace oneflow {

CnclSubTskGphBuilder::CnclSubTskGphBuilder() {
  std::vector<std::shared_ptr<SubTskGphBuilder>> builders;

  builders.emplace_back(new CclAllReduceSubTskGphBuilder(DeviceType::kMLU));
  builders.emplace_back(new CclReduceScatterSubTskGphBuilder(DeviceType::kMLU));
  builders.emplace_back(new CclP2SNoncontinuousSubTskGphBuilder(DeviceType::kMLU));
  builders.emplace_back(new CclAllGatherSubTskGphBuilder(DeviceType::kMLU));
  builders.emplace_back(new CclS2BNoncontinuousSubTskGphBuilder(DeviceType::kMLU));
  builders.emplace_back(new CclReduceSubTskGphBuilder(DeviceType::kMLU));
  builders.emplace_back(new CclScatterThenAllGatherSubTskGphBuilder(DeviceType::kMLU));
  builders.emplace_back(new CclBroadcastSubTskGphBuilder(DeviceType::kMLU));
  builders.emplace_back(new CclAll2AllSubTskGphBuilder(DeviceType::kMLU));

  chain_builder_.reset(new ChainSubTskGphBuilder(builders));
}

Maybe<SubTskGphBuilderStatus> CnclSubTskGphBuilder::Build(
    SubTskGphBuilderCtx* ctx, const std::vector<TaskNode*>& sorted_in_tasks,
    std::vector<TaskNode*>* sorted_out_tasks,
    std::vector<std::vector<TaskNode*>>* sorted_ctrl_tasks, const ParallelDesc& in_parallel_desc,
    const ParallelDesc& out_parallel_desc, const LogicalBlobId& lbi,
    const BlobDesc& logical_blob_desc, const SbpParallel& in_sbp_parallel,
    const SbpParallel& out_sbp_parallel, const Shape& time_shape) const {
  if (!GlobalJobDesc().Bool("__is_user_function__")) { return Error::BoxingNotSupportedError(); }
  if (!IsSourceTimeShape(time_shape)) { return Error::BoxingNotSupportedError(); }
  return chain_builder_->Build(ctx, sorted_in_tasks, sorted_out_tasks, sorted_ctrl_tasks,
                               in_parallel_desc, out_parallel_desc, lbi, logical_blob_desc,
                               in_sbp_parallel, out_sbp_parallel, time_shape);
}

}  // namespace oneflow
