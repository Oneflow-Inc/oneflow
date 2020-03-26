#include <cstdint>
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

class GenerateRandomBatchPermutationIndicesCPUKernel final : public user_op::OpKernel {
 public:
  GenerateRandomBatchPermutationIndicesCPUKernel(user_op::KernelInitContext* ctx)
      : user_op::OpKernel(ctx) {
    int64_t seed = ctx->GetAttr<int64_t>("seed");
    random_generator_.reset(new std::mt19937(seed));
  }

  GenerateRandomBatchPermutationIndicesCPUKernel() = default;
  ~GenerateRandomBatchPermutationIndicesCPUKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    std::iota(y->mut_dptr<int32_t>(), y->mut_dptr<int32_t>() + y->shape().elem_cnt(), 0);
    std::shuffle(y->mut_dptr<int32_t>(), y->mut_dptr<int32_t>() + y->shape().elem_cnt(),
                 *random_generator_);
  };

  std::unique_ptr<std::mt19937> random_generator_;
};

REGISTER_USER_KERNEL("generate_random_batch_permutation_indices")
    .SetCreateFn([](oneflow::user_op::KernelInitContext* ctx) {
      return new GenerateRandomBatchPermutationIndicesCPUKernel(ctx);
    })
    .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {
      return ctx.device_type() == DeviceType::kCPU;
    });

}  // namespace oneflow
