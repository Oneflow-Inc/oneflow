#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/utils/pool_util.h"

namespace oneflow {

template<typename T>
class GPUMaxpool2dKernel final : public user_op::OpKernel {
 public:
  GPUMaxpool2dKernel() = default;
  ~GPUMaxpool2dKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // add your code here
  };
};

#define REGISTER_GPU_MAX_POOL_2D_KERNEL(dtype)                           \
  REGISTER_USER_KERNEL("max_pool_2d")                                          \
      .SetCreateFn<GPUMaxpool2dKernel<dtype>>()                              \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                  \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0); \
        return ctx.device_type() == DeviceType::kGPU                                \
               && y_desc->data_type() == GetDataType<dtype>::value;                 \
      });

REGISTER_GPU_MAX_POOL_2D_KERNEL(float)
REGISTER_GPU_MAX_POOL_2D_KERNEL(double)

template<typename T>
class GpuMaxpool2dGradKernel final : public user_op::OpKernel {
 public:
  GpuMaxpool2dGradKernel() = default;
  ~GpuMaxpool2dGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // add your code
  };
};

#define REGISTER_GPU_MAX_POOL_2D_GRAD_KERNEL(dtype)                             \
  REGISTER_USER_KERNEL("max_pool_2d_grad")                                       \
      .SetCreateFn<GpuMaxpool2dGradKernel<dtype>>()                            \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0); \
        return ctx.device_type() == DeviceType::kGPU                                  \
               && dx_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_GPU_MAX_POOL_2D_GRAD_KERNEL(float)
REGISTER_GPU_MAX_POOL_2D_GRAD_KERNEL(double)

}  // namespace oneflow
