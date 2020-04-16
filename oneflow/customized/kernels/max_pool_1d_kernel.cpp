#include "oneflow/core/framework/framework.h"
#include "oneflow/customized/utils/pool_util.h"

namespace oneflow {

template<typename T>
class CPUMaxpool1dKernel final : public user_op::OpKernel {
 public:
  CPUMaxpool1dKernel() = default;
  ~CPUMaxpool1dKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // add your code here
  };
};

#define REGISTER_CPU_MAX_POOL_1D_KERNEL(dtype)                           \
  REGISTER_USER_KERNEL("max_pool_1d")                                          \
      .SetCreateFn<CPUMaxpool1dKernel<dtype>>()                              \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                  \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0); \
        return ctx.device_type() == DeviceType::kCPU                                \
               && y_desc->data_type() == GetDataType<dtype>::value;                 \
      });

REGISTER_CPU_MAX_POOL_1D_KERNEL(float)
REGISTER_CPU_MAX_POOL_1D_KERNEL(double)

template<typename T>
class CpuMaxpool1dGradKernel final : public user_op::OpKernel {
 public:
  CpuMaxpool1dGradKernel() = default;
  ~CpuMaxpool1dGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    // add your code
  };
};

#define REGISTER_CPU_MAX_POOL_1D_GRAD_KERNEL(dtype)                             \
  REGISTER_USER_KERNEL("max_pool_1d_grad")                                       \
      .SetCreateFn<CpuMaxpool1dGradKernel<dtype>>()                            \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0); \
        return ctx.device_type() == DeviceType::kCPU                                  \
               && dx_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_CPU_MAX_POOL_1D_GRAD_KERNEL(float)
REGISTER_CPU_MAX_POOL_1D_GRAD_KERNEL(double)

}  // namespace oneflow
