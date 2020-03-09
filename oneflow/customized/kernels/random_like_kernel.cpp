#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/random_generator.h"

namespace oneflow {

template<typename T>
class RandomLikeKernel final : public user_op::OpKernel {
 public:
  RandomLikeKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  RandomLikeKernel() = default;
  ~RandomLikeKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    // TODO: tsai, chengcheng do the initialization in kernel init when interface ready
    if (random_generator_.get() == nullptr) {
      int64_t seed = GetCurTime();
      const bool has_seed = ctx->GetAttr<int32_t>("has_seed") == 1;
      if (has_seed) { seed = ctx->GetAttr<int64_t>("seed"); }
      random_generator_.reset(new RandomGenerator<DeviceType::kGPU>(seed, ctx->device_ctx()));
    }
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    random_generator_->Uniform(out->shape().elem_cnt(), out->mut_dptr<T>());
  };
  std::unique_ptr<RandomGenerator<DeviceType::kGPU>> random_generator_;
};

#define REGISTER_RANDOM_LIKE_KERNEL(dtype, dev)                                                \
  REGISTER_USER_KERNEL("random_like")                                                          \
      .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {                        \
        return new RandomLikeKernel<dtype>(ctx);                                               \
      })                                                                                       \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0);        \
        return ctx.device_type() == dev && out_desc->data_type() == GetDataType<dtype>::value; \
      });

REGISTER_RANDOM_LIKE_KERNEL(float, DeviceType::kGPU)

}  // namespace oneflow
