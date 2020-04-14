#include "oneflow/core/framework/framework.h"

namespace oneflow {

template<typename T>
class CPUPoolKernel final : public user_op::OpKernel {
 public:
  CPUPoolKernel() = default;
  ~CPUPoolKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override{
      // add your code here
  };
};

#define REGISTER_CPU_POOL_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("pool").SetCreateFn<CPUPoolKernel<dtype>>().SetIsMatchedPred( \
      [](const user_op::KernelRegContext& ctx) {                                     \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);  \
        return ctx.device_type() == DeviceType::kCPU                                 \
               && y_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_CPU_POOL_KERNEL(float)
REGISTER_CPU_POOL_KERNEL(double)

template<typename T>
class CpuPoolGradKernel final : public user_op::OpKernel {
 public:
  CpuPoolGradKernel() = default;
  ~CpuPoolGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override{
      // add your code
  };
};

#define REGISTER_CPU_POOL_KERNEL(dtype)                                               \
  REGISTER_USER_KERNEL("pool_grad")                                                   \
      .SetCreateFn<CpuPoolGradKernel<dtype>>()                                        \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0); \
        return ctx.device_type() == DeviceType::kCPU                                  \
               && dx_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_CPU_POOL_KERNEL(float)
REGISTER_CPU_POOL_KERNEL(double)

}  // namespace oneflow
