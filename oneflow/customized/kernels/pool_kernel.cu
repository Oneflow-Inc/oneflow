#include "oneflow/core/framework/framework.h"

namespace oneflow {

template<typename T>
class GPUPoolKernel final : public user_op::OpKernel {
 public:
  GPUPoolKernel() = default;
  ~GPUPoolKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override{
      // add your code here
  };
};

#define REGISTER_GPU_POOL_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("pool").SetCreateFn<GPUPoolKernel<dtype>>().SetIsMatchedPred( \
      [](const user_op::KernelRegContext& ctx) {                                     \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);  \
        return ctx.device_type() == DeviceType::kGPU                                 \
               && y_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_GPU_POOL_KERNEL(float)
REGISTER_GPU_POOL_KERNEL(double)

template<typename T>
class GpuPoolGradKernel final : public user_op::OpKernel {
 public:
  GpuPoolGradKernel() = default;
  ~GpuPoolGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override{
      // add your code
  };
};

#define REGISTER_GPU_POOL_KERNEL(dtype)                                               \
  REGISTER_USER_KERNEL("pool_grad")                                                   \
      .SetCreateFn<GpuPoolGradKernel<dtype>>()                                        \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0); \
        return ctx.device_type() == DeviceType::kGPU                                  \
               && dx_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_GPU_POOL_KERNEL(float)
REGISTER_GPU_POOL_KERNEL(double)

}  // namespace oneflow
