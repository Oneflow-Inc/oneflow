#include <cstdint>
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

class GenerateRandomBatchPermutationIndicesCPUKernel final : public user_op::OpKernel {
 public:
  GenerateRandomBatchPermutationIndicesCPUKernel() = default;
  ~GenerateRandomBatchPermutationIndicesCPUKernel() = default;

 private:
  void NewOpKernelContext(user_op::KernelInitContext* ctx,
                          user_op::OpKernelContext** opkernel_ctx) const override {
    int64_t seed = ctx->GetAttr<int64_t>("seed");
    *opkernel_ctx = new user_op::OpKernelContextIf<std::mt19937>(seed);
  }
  void Compute(user_op::KernelComputeContext* ctx,
               user_op::OpKernelContext* opkernel_ctx) const override {
    auto* random_generator = dynamic_cast<user_op::OpKernelContextIf<std::mt19937>*>(opkernel_ctx);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    std::iota(y->mut_dptr<int32_t>(), y->mut_dptr<int32_t>() + y->shape().elem_cnt(), 0);
    std::shuffle(y->mut_dptr<int32_t>(), y->mut_dptr<int32_t>() + y->shape().elem_cnt(),
                 *random_generator->Mutable());
  };
};

REGISTER_USER_KERNEL("generate_random_batch_permutation_indices")
    .SetCreateFn<GenerateRandomBatchPermutationIndicesCPUKernel>()
    .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {
      return ctx.device_type() == DeviceType::kCPU;
    });

}  // namespace oneflow
