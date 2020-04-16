#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/utils/pool_util.h"

namespace oneflow {

template<typename T>
class CPUAvgpool2dKernel final : public user_op::OpKernel {
 public:
  CPUAvgpool2dKernel() = default;
  ~CPUAvgpool2dKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // add your code here
  };
};

#define REGISTER_CPU_AVG_POOL_2D_KERNEL(dtype)                           \
  REGISTER_USER_KERNEL("avg_pool_2d")                                          \
      .SetCreateFn<CPUAvgpool2dKernel<dtype>>()                              \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                  \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0); \
        return ctx.device_type() == DeviceType::kCPU                                \
               && y_desc->data_type() == GetDataType<dtype>::value;                 \
      });

REGISTER_CPU_AVG_POOL_2D_KERNEL(float)
REGISTER_CPU_AVG_POOL_2D_KERNEL(double)

template<typename T>
class CpuAvgpool2dGradKernel final : public user_op::OpKernel {
 public:
  CpuAvgpool2dGradKernel() = default;
  ~CpuAvgpool2dGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // add your code
  };
};

#define REGISTER_CPU_AVG_POOL_2D_GRAD_KERNEL(dtype)                             \
  REGISTER_USER_KERNEL("avg_pool_2d_grad")                                       \
      .SetCreateFn<CpuAvgpool2dGradKernel<dtype>>()                            \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0); \
        return ctx.device_type() == DeviceType::kCPU                                  \
               && dx_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_CPU_AVG_POOL_2D_GRAD_KERNEL(float)
REGISTER_CPU_AVG_POOL_2D_GRAD_KERNEL(double)

}  // namespace oneflow
