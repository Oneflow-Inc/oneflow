#include "oneflow/core/framework/framework.h"

namespace oneflow {

template<typename T>
class LayerNormCpuKernel final : public user_op::OpKernel {
 public:
  LayerNormCpuKernel() = default;
  ~LayerNormCpuKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override { TODO(); };
};

#define REGISTER_LAYER_NORM_CPU_KERNEL(dtype)                                       \
  REGISTER_USER_KERNEL("layer_norm")                                                \
      .SetCreateFn<LayerNormCpuKernel<dtype>>()                                     \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                  \
        const user_op::TensorDesc* x_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0); \
        return ctx.device_type() == DeviceType::kCPU                                \
               && x_desc->data_type() == GetDataType<dtype>::value;                 \
      });

REGISTER_LAYER_NORM_CPU_KERNEL(float)
REGISTER_LAYER_NORM_CPU_KERNEL(double)

template<typename T>
class LayerNormGradCpuKernel final : public user_op::OpKernel {
 public:
  LayerNormGradCpuKernel() = default;
  ~LayerNormGradCpuKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override { TODO(); };
};

#define REGISTER_LAYER_NORM_GRAD_CPU_KERNEL(dtype)                                    \
  REGISTER_USER_KERNEL("layer_norm_grad")                                             \
      .SetCreateFn<LayerNormGradCpuKernel<dtype>>()                                   \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dy_desc = ctx.TensorDesc4ArgNameAndIndex("dy", 0); \
        return ctx.device_type() == DeviceType::kCPU                                  \
               && dy_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_LAYER_NORM_GRAD_CPU_KERNEL(float)
REGISTER_LAYER_NORM_GRAD_CPU_KERNEL(double)

template<typename T>
class LayerNormParamGradCpuKernel final : public user_op::OpKernel {
 public:
  LayerNormParamGradCpuKernel() = default;
  ~LayerNormParamGradCpuKernel() = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
  void Compute(user_op::KernelComputeContext* ctx) const override { TODO(); };
};

#define REGISTER_LAYER_NORM_PARAM_GRAD_CPU_KERNEL(dtype)                              \
  REGISTER_USER_KERNEL("layer_norm_param_grad")                                       \
      .SetCreateFn<LayerNormParamGradCpuKernel<dtype>>()                              \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dy_desc = ctx.TensorDesc4ArgNameAndIndex("dy", 0); \
        return ctx.device_type() == DeviceType::kCPU                                  \
               && dy_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_LAYER_NORM_PARAM_GRAD_CPU_KERNEL(float)
REGISTER_LAYER_NORM_PARAM_GRAD_CPU_KERNEL(double)

}  // namespace oneflow
