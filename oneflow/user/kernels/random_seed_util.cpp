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
#include "oneflow/user/kernels/random_seed_util.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {

Maybe<uint64_t> GetOpKernelRandomSeed(const user_op::KernelInitContext* ctx) {
  int64_t seed = ctx->Attr<int64_t>("seed");
  if (!ctx->Attr<bool>("has_seed")) { seed = NewRandomSeed(); }
  return GetOpKernelRandomSeedInCurrentRank(ctx, seed);
}

Maybe<uint64_t> GetRandomSeedForRank(const ParallelDesc& placement, const NdSbp& nd_sbp,
                                     uint64_t init_seed, int64_t rank_id) {
  uint64_t seed = init_seed;
  const Shape& hierarchy = *placement.hierarchy();
  std::vector<int64_t> coordinate(hierarchy.NumAxes());
  int64_t seed_idx = 0;
  int64_t stride = 1;
  for (int i = nd_sbp.sbp_parallel_size() - 1; i >= 0; --i) {
    // coordinate at axis i
    int coord = rank_id % hierarchy.At(i);
    rank_id = (rank_id - coord) / hierarchy.At(i);
    // coordinate reset to 0 if broadcast
    if (nd_sbp.sbp_parallel(i).has_broadcast_parallel()) {
      // do nothing
    } else if (nd_sbp.sbp_parallel(i).has_split_parallel()) {
      seed_idx += coord * stride;
      stride *= hierarchy.At(i);
    } else {
      // other sbp is not allowed
      return Error::RuntimeError() << "random source op only support broadcast or split";
    }
  }
  std::seed_seq seq{init_seed};
  std::vector<uint64_t> seeds(stride);
  seq.generate(seeds.begin(), seeds.end());
  seed = JUST(VectorAt(seeds, seed_idx));
  return seed;
}

Maybe<uint64_t> GetOpKernelRandomSeedInCurrentRank(const user_op::KernelInitContext* ctx,
                                                   uint64_t init_seed) {
  if (ctx->parallel_ctx().parallel_num() == 1) { return init_seed; }
  const auto& outputs = ctx->outputs();
  CHECK_EQ(outputs.size(), 1);
  return GetRandomSeedForRank(ctx->parallel_desc(), ctx->NdSbp4ArgNameAndIndex("out", 0), init_seed,
                              ctx->parallel_ctx().parallel_id());
}

}  // namespace oneflow
