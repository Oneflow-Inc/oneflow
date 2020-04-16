#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/utils/pool_util.h"

namespace oneflow {

template<typename T>
class GPUAvgpool1dKernel final : public user_op::OpKernel {
 public:
  GPUAvgpool1dKernel() = default;
  ~GPUAvgpool1dKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // add your code here
  };
};

#define REGISTER_GPU_AVG_POOL_1D_KERNEL(dtype)                           \
  REGISTER_USER_KERNEL("avg_pool_1d")                                          \
      .SetCreateFn<GPUAvgpool1dKernel<dtype>>()                              \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                  \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0); \
        return ctx.device_type() == DeviceType::kGPU                                \
               && y_desc->data_type() == GetDataType<dtype>::value;                 \
      });

REGISTER_GPU_AVG_POOL_1D_KERNEL(float)
REGISTER_GPU_AVG_POOL_1D_KERNEL(double)

template<typename T>
class GpuAvgpool1dGradKernel final : public user_op::OpKernel {
 public:
  GpuAvgpool1dGradKernel() = default;
  ~GpuAvgpool1dGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // add your code
  };
};

#define REGISTER_GPU_AVG_POOL_1D_GRAD_KERNEL(dtype)                             \
  REGISTER_USER_KERNEL("avg_pool_1d_grad")                                       \
      .SetCreateFn<GpuAvgpool1dGradKernel<dtype>>()                            \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0); \
        return ctx.device_type() == DeviceType::kGPU                                  \
               && dx_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_GPU_AVG_POOL_1D_GRAD_KERNEL(float)
REGISTER_GPU_AVG_POOL_1D_GRAD_KERNEL(double)

}  // namespace oneflow
