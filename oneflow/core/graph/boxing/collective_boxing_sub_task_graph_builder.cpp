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
#include "oneflow/core/graph/boxing/collective_boxing_sub_task_graph_builder.h"
#include "oneflow/core/graph/boxing/ccl_sub_task_graph_builder.h"

namespace oneflow {

CollectiveBoxingSubTskGphBuilder::CollectiveBoxingSubTskGphBuilder() {
  const CollectiveBoxingConf collective_boxing_conf =
      Singleton<ResourceDesc, ForSession>::Get()->collective_boxing_conf();
  std::vector<std::shared_ptr<SubTskGphBuilder>> builders;
  builders.emplace_back(new CclAllReduceSubTskGphBuilder(DeviceType::kCUDA));
  builders.emplace_back(new CclReduceScatterSubTskGphBuilder(DeviceType::kCUDA));
  builders.emplace_back(new CclP2SNoncontinuousSubTskGphBuilder(DeviceType::kCUDA));
  builders.emplace_back(new CclAllGatherSubTskGphBuilder(DeviceType::kCUDA));
  builders.emplace_back(new CclS2BNoncontinuousSubTskGphBuilder(DeviceType::kCUDA));
  builders.emplace_back(new CclReduceSubTskGphBuilder(DeviceType::kCUDA));
  builders.emplace_back(new CclScatterThenAllGatherSubTskGphBuilder(DeviceType::kCUDA));
  builders.emplace_back(new CclBroadcastSubTskGphBuilder(DeviceType::kCUDA));

  if (collective_boxing_conf.nccl_enable_all_to_all()) {
#if defined(WITH_CUDA) && NCCL_VERSION_CODE > 2700
    builders.emplace_back(new CclAll2AllSubTskGphBuilder(DeviceType::kCUDA));
#else
    LOG(WARNING) << "nccl_enable_all_to_all is unavailable unless NCCL_VERSION > 2.7.0";
#endif
  }
  chain_builder_.reset(new ChainSubTskGphBuilder(builders));
}

Maybe<SubTskGphBuilderStatus> CollectiveBoxingSubTskGphBuilder::Build(
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
