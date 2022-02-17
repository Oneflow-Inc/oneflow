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
    const SbpParallel& out_sbp =
        ctx->SbpParallel4ArgNameAndIndex(outputs.at(0).first, outputs.at(0).second);
    if (out_sbp.has_split_parallel()) {
      std::seed_seq seq{seed};
      std::vector<int64_t> seeds(parallel_num);
      seq.generate(seeds.begin(), seeds.end());
      seed = seeds.at(ctx->parallel_ctx().parallel_id());
    } else {
      CHECK(out_sbp.has_broadcast_parallel());
    }
  }
  return seed;
}

}  // namespace oneflow
