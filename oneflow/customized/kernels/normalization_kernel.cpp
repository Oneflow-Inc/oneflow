#include "oneflow/core/framework/framework.h"

namespace oneflow {

template<typename T>
class CpuNormalizationKernel final : public user_op::OpKernel {
 public:
  CpuNormalizationKernel() = default;
  ~CpuNormalizationKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    UNIMPLEMENTED();
  };
};

#define REGISTER_CPU_KERNEL(dtype)                                                 \
  REGISTER_USER_KERNEL("normalization").SetCreateFn<CpuNormalizationKernel<dtype>>().SetIsMatchedPred(    \
      [](const user_op::KernelRegContext& ctx) {                                        \
        const user_op::TensorDesc* out_desc = ctx.TensorDesc4ArgNameAndIndex("out", 0); \
        return ctx.device_type() == DeviceType::kCPU                                    \
               && out_desc->data_type() == GetDataType<dtype>::value;                   \
      });

REGISTER_CPU_KERNEL(float)
REGISTER_CPU_KERNEL(double)

template<typename T>
class CpuNormalizationGradKernel final : public user_op::OpKernel {
 public:
  CpuNormalizationGradKernel() = default;
  ~CpuNormalizationGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    UNIMPLEMENTED();
  };
};

#define REGISTER_CPU_GRAD_KERNEL(dtype)                                          \
  REGISTER_USER_KERNEL("normalization_grad")                                                   \
      .SetCreateFn<CpuNormalizationGradKernel<dtype>>()                                        \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0); \
        return ctx.device_type() == DeviceType::kCPU                                  \
               && dx_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_CPU_GRAD_KERNEL(float)
REGISTER_CPU_GRAD_KERNEL(double)

}  // namespace oneflow

