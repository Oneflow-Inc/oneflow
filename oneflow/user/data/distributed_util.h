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
#ifndef ONEFLOW_USER_DATA_DISTRIBUTED_UTIL_H_
#define ONEFLOW_USER_DATA_DISTRIBUTED_UTIL_H_

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/job/sbp_parallel.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

namespace data {

inline Maybe<void> InitDataSourceDistributedInfo(user_op::KernelInitContext* ctx,
                                                 size_t& world_size, int64_t& rank) {
  auto nd_sbp_str_vec = ctx->Attr<std::vector<std::string>>("nd_sbp");
  if (nd_sbp_str_vec.empty()) {
    world_size = GlobalProcessCtx::WorldSize();
    rank = GlobalProcessCtx::Rank();
  } else {
    const Shape& hierarchy = *ctx->parallel_desc().hierarchy();
    CHECK_EQ_OR_RETURN(hierarchy.NumAxes(), nd_sbp_str_vec.size());
    rank = 0;
    world_size = 1;

    using index_helper_t = NdIndexOffsetHelper<int64_t, SHAPE_MAX_AXIS_SIZE>;
    index_helper_t index_helper(hierarchy.dim_vec().data(), hierarchy.NumAxes());
    int64_t nd_index[SHAPE_MAX_AXIS_SIZE] = {0};
    index_helper.OffsetToNdIndex(ctx->parallel_ctx().parallel_id(), nd_index);

    for (int i = hierarchy.NumAxes() - 1; i >= 0; --i) {
      SbpParallel sbp;
      CHECK_OR_RETURN(ParseSbpParallelFromString(nd_sbp_str_vec[i], &sbp));
      if (sbp.has_split_parallel()) {
        rank += nd_index[i] * world_size;
        world_size *= hierarchy.At(i);
      }
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace data

}  // namespace oneflow

#endif  // ONEFLOW_CUSTOMIZED_DATA_ONEREC_DATA_READER_H_
