#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/utils/pool_util.h"

namespace oneflow {

template<typename T>
class CPUAvgpool3dKernel final : public user_op::OpKernel {
 public:
  CPUAvgpool3dKernel() = default;
  ~CPUAvgpool3dKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // add your code here
  };
};

#define REGISTER_CPU_AVG_POOL_3D_KERNEL(dtype)                           \
  REGISTER_USER_KERNEL("avg_pool_3d")                                          \
      .SetCreateFn<CPUAvgpool3dKernel<dtype>>()                              \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                  \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0); \
        return ctx.device_type() == DeviceType::kCPU                                \
               && y_desc->data_type() == GetDataType<dtype>::value;                 \
      });

REGISTER_CPU_AVG_POOL_3D_KERNEL(float)
REGISTER_CPU_AVG_POOL_3D_KERNEL(double)

template<typename T>
class CpuAvgpool3dGradKernel final : public user_op::OpKernel {
 public:
  CpuAvgpool3dGradKernel() = default;
  ~CpuAvgpool3dGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // add your code
  };
};

#define REGISTER_CPU_AVG_POOL_3D_GRAD_KERNEL(dtype)                             \
  REGISTER_USER_KERNEL("avg_pool_3d_grad")                                       \
      .SetCreateFn<CpuAvgpool3dGradKernel<dtype>>()                            \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0); \
        return ctx.device_type() == DeviceType::kCPU                                  \
               && dx_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_CPU_AVG_POOL_3D_GRAD_KERNEL(float)
REGISTER_CPU_AVG_POOL_3D_GRAD_KERNEL(double)

}  // namespace oneflow
