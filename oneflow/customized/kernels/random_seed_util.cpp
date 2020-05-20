#include "oneflow/customized/kernels/random_seed_util.h"

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
