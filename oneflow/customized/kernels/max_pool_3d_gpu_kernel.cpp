#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/utils/pool_util.h"

namespace oneflow {

template<typename T>
class GPUMaxpool3dKernel final : public user_op::OpKernel {
 public:
  GPUMaxpool3dKernel() = default;
  ~GPUMaxpool3dKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // add your code here
  };
};

#define REGISTER_GPU_MAX_POOL_3D_KERNEL(dtype)                           \
  REGISTER_USER_KERNEL("max_pool_3d")                                          \
      .SetCreateFn<GPUMaxpool3dKernel<dtype>>()                              \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                  \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0); \
        return ctx.device_type() == DeviceType::kGPU                                \
               && y_desc->data_type() == GetDataType<dtype>::value;                 \
      });

REGISTER_GPU_MAX_POOL_3D_KERNEL(float)
REGISTER_GPU_MAX_POOL_3D_KERNEL(double)

template<typename T>
class GpuMaxpool3dGradKernel final : public user_op::OpKernel {
 public:
  GpuMaxpool3dGradKernel() = default;
  ~GpuMaxpool3dGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // add your code
  };
};

#define REGISTER_GPU_MAX_POOL_3D_GRAD_KERNEL(dtype)                             \
  REGISTER_USER_KERNEL("max_pool_3d_grad")                                       \
      .SetCreateFn<GpuMaxpool3dGradKernel<dtype>>()                            \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0); \
        return ctx.device_type() == DeviceType::kGPU                                  \
               && dx_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_GPU_MAX_POOL_3D_GRAD_KERNEL(float)
REGISTER_GPU_MAX_POOL_3D_GRAD_KERNEL(double)

}  // namespace oneflow
