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

namespace oneflow {

int64_t GetOpKernelRandomSeed(const user_op::KernelInitContext* ctx) {
  int64_t seed = ctx->Attr<int64_t>("seed");
  if (!ctx->Attr<bool>("has_seed")) { seed = NewRandomSeed(); }

  int64_t parallel_num = ctx->parallel_ctx().parallel_num();
  const auto& outputs = ctx->outputs();
  CHECK_EQ(outputs.size(), 1);

  if (parallel_num > 1) {
    int64_t seed_num = 1;
    const Shape& hierarchy = *ctx->parallel_desc().hierarchy();
    int64_t parallel_id = ctx->parallel_ctx().parallel_id();
    const NdSbp& nd_sbp = ctx->NdSbp4ArgNameAndIndex(outputs.at(0).first, outputs.at(0).second);
    std::vector<int64_t> coordinate(hierarchy.NumAxes());

    for (int32_t i = 0; i < nd_sbp.sbp_parallel_size(); ++i) {
      // parallel coordinate
      parallel_num /= hierarchy.At(i);
      coordinate.at(i) = parallel_id / parallel_num;
      parallel_id %= parallel_num;
      // seed number
      if (nd_sbp.sbp_parallel(i).has_split_parallel()) {
        seed_num *= hierarchy.At(i);
      } else {
        CHECK(nd_sbp.sbp_parallel(i).has_broadcast_parallel());
      }
    }
    // Get seed index
    int seed_idx = 0;
    int weight = 1;
    for (int32_t i = nd_sbp.sbp_parallel_size() - 1; i >= 0; --i) {
      if (nd_sbp.sbp_parallel(i).has_split_parallel()) {
        seed_idx += weight * coordinate.at(i);
        weight *= hierarchy.At(i);
      }
    }
    std::seed_seq seq{seed};
    std::vector<int64_t> seeds(seed_num);
    seq.generate(seeds.begin(), seeds.end());
    seed = seeds.at(seed_idx);
  }
  return seed;
}

}  // namespace oneflow
