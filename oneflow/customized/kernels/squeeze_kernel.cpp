#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<typename T, DeviceType device_type>
class SqueezeKernel final : public user_op::OpKernel {
 public:
  SqueezeKernel(const user_op::KernelInitContext& ctx) : user_op::OpKernel(ctx) {}
  SqueezeKernel() = default;
  ~SqueezeKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    Memcpy<device_type>(ctx->device_ctx(), out->mut_dptr<T>(), in->dptr<T>(),
                        in->shape().elem_cnt() * sizeof(T));
  };
};

#define REGISTER_SQUEEZE_KERNEL(dtype)                                      \
  REGISTER_USER_KERNEL("squeeze")                                           \
      .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {     \
        return new SqueezeKernel<dtype, DeviceType::kCPU>(ctx);             \
      })                                                                    \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) { \
        return ctx.device_type() == DeviceType::kCPU;                       \
      });                                                                   \
  REGISTER_USER_KERNEL("squeeze")                                           \
      .SetCreateFn([](const oneflow::user_op::KernelInitContext& ctx) {     \
        return new SqueezeKernel<dtype, DeviceType::kGPU>(ctx);             \
      })                                                                    \
      .SetIsMatchedPred([](const oneflow::user_op::KernelRegContext& ctx) { \
        return ctx.device_type() == DeviceType::kGPU;                       \
      });

REGISTER_SQUEEZE_KERNEL(float)
REGISTER_SQUEEZE_KERNEL(double)
REGISTER_SQUEEZE_KERNEL(int32_t)
REGISTER_SQUEEZE_KERNEL(int64_t)

}  // namespace oneflow
