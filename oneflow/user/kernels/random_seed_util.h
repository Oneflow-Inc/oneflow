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
#ifndef ONEFLOW_USER_KERNELS_RANDOM_SEED_UTIL_H_
#define ONEFLOW_USER_KERNELS_RANDOM_SEED_UTIL_H_

#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/core/framework/random_generator.h"

namespace oneflow {

Maybe<uint64_t> GetRandomSeedForRank(const ParallelDesc& placement, const NdSbp& nd_sbp,
                                     uint64_t init_seed, int64_t rank_id);

Maybe<uint64_t> GetOpKernelRandomSeed(const user_op::KernelInitContext* ctx);
Maybe<uint64_t> GetOpKernelRandomSeedInCurrentRank(const user_op::KernelInitContext* ctx,
                                                   uint64_t init_seed,
                                                   const user_op::OpArg& arg = {"out", 0});

Maybe<one::Generator> GetGeneratorForLazyOrGlobal(const std::shared_ptr<one::Generator>& generator,
                                                  bool is_lazy,
                                                  const Optional<Symbol<ParallelDesc>>& placement,
                                                  const Optional<Symbol<NdSbp>>& nd_sbp);

Maybe<one::Generator> GetGeneratorForLazyOrGlobal(const std::shared_ptr<one::Generator>& generator,
                                                  bool is_lazy,
                                                  const std::shared_ptr<one::Tensor>& input);

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_RANDOM_SEED_UTIL_H_
