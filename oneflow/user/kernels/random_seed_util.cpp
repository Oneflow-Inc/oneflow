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
#include "oneflow/core/functional/impl/common.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

namespace oneflow {

Maybe<uint64_t> GetOpKernelRandomSeed(const user_op::KernelInitContext* ctx) {
  int64_t seed = ctx->Attr<int64_t>("seed");
  if (!ctx->Attr<bool>("has_seed")) { seed = NewRandomSeed(); }
  return GetOpKernelRandomSeedInCurrentRank(ctx, seed);
}

// NOTE: Get random seed in current rank, and ensure that it will have same seed between
// broadcast sbp and it will be different between split sbp.
//
// It will scan nd_sbp from last axis to first axis(It likes the algorithm in NdIndexOffsetHelper).
// If sbp is broadcast, this axis will skip.
// If sbp is split, it will use rand_id to accumulate the offset.
Maybe<uint64_t> GetRandomSeedForRank(const ParallelDesc& placement, const NdSbp& nd_sbp,
                                     uint64_t init_seed, int64_t rank_id) {
  uint64_t seed = init_seed;
  const Shape& hierarchy = *placement.hierarchy();
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
                                                   uint64_t init_seed, const user_op::OpArg& arg) {
  if (ctx->parallel_ctx().parallel_num() == 1) { return init_seed; }
  CHECK_OR_RETURN(ctx->has_output(arg.name(), arg.index()))
      << arg.name() << "_" << arg.index() << " not exist";
  const auto& nd_sbp = ctx->NdSbp4ArgNameAndIndex(arg.name(), arg.index());
  return GetRandomSeedForRank(ctx->parallel_desc(), nd_sbp, init_seed,
                              ctx->parallel_ctx().parallel_id());
}

Maybe<one::Generator> GetGeneratorForLazyOrGlobal(const std::shared_ptr<one::Generator>& generator,
                                                  bool is_lazy,
                                                  const Optional<Symbol<ParallelDesc>>& placement,
                                                  const Optional<Symbol<NdSbp>>& nd_sbp) {
  bool is_global = placement.has_value() && nd_sbp.has_value();
  if (!is_lazy && !is_global) { return generator; }

  auto cpu_gen_impl = JUST(generator->Get<one::CPUGeneratorImpl>(0));
  CHECK_OR_RETURN(cpu_gen_impl) << "expect a CPUGeneratorImpl";
  uint64_t init_seed = cpu_gen_impl->engine()();
  auto new_gen = JUST(one::MakeGenerator(JUST(generator->device())->type()));
  if (is_lazy) {
    new_gen->set_current_seed(init_seed);
    return new_gen;
  }

  uint64_t rank_seed = init_seed;
  if (JUST(placement)->parallel_num() > 1) {
    JUST(one::functional::BroadcastSeedToAllRanks(&init_seed, /*root=*/0));
    rank_seed = JUST(
        GetRandomSeedForRank(*JUST(placement), *JUST(nd_sbp), init_seed, GlobalProcessCtx::Rank()));
  }
  new_gen->set_current_seed(rank_seed);
  return new_gen;
}

Maybe<one::Generator> GetGeneratorForLazyOrGlobal(const std::shared_ptr<one::Generator>& generator,
                                                  bool is_lazy,
                                                  const std::shared_ptr<one::Tensor>& input) {
  if (input->is_global()) {
    return GetGeneratorForLazyOrGlobal(generator, is_lazy, JUST(input->parallel_desc()),
                                       JUST(input->nd_sbp()));
  } else {
    return GetGeneratorForLazyOrGlobal(generator, is_lazy, NullOpt, NullOpt);
  }
}

}  // namespace oneflow
