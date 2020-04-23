#include "oneflow/core/framework/framework.h"

namespace oneflow {

template<typename T>
class LayerNormGpuKernel final : public user_op::OpKernel {
 public:
  LayerNormGpuKernel() = default;
  ~LayerNormGpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override{
      // Add your code...
  };
};

#define REGISTER_LAYER_NORM_GPU_KERNEL(dtype)                                       \
  REGISTER_USER_KERNEL("layer_norm")                                                \
      .SetCreateFn<LayerNormGpuKernel<dtype>>()                                     \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                  \
        const user_op::TensorDesc* x_desc = ctx.TensorDesc4ArgNameAndIndex("x", 0); \
        return ctx.device_type() == DeviceType::kGPU                                \
               && x_desc->data_type() == GetDataType<dtype>::value;                 \
      });

REGISTER_LAYER_NORM_GPU_KERNEL(float)
REGISTER_LAYER_NORM_GPU_KERNEL(double)

template<typename T>
class LayerNormGradGpuKernel final : public user_op::OpKernel {
 public:
  LayerNormGradGpuKernel() = default;
  ~LayerNormGradGpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override{
      // Add your code...
  };
};

#define REGISTER_LAYER_NORM_GRAD_GPU_KERNEL(dtype)                                    \
  REGISTER_USER_KERNEL("layer_norm_grad")                                             \
      .SetCreateFn<LayerNormGradGpuKernel<dtype>>()                                   \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dy_desc = ctx.TensorDesc4ArgNameAndIndex("dy", 0); \
        return ctx.device_type() == DeviceType::kGPU                                  \
               && dy_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_LAYER_NORM_GRAD_GPU_KERNEL(float)
REGISTER_LAYER_NORM_GRAD_GPU_KERNEL(double)

template<typename T>
class LayerNormParamGradGpuKernel final : public user_op::OpKernel {
 public:
  LayerNormParamGradGpuKernel() = default;
  ~LayerNormParamGradGpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override{
      // Add your code...
  };
};

#define REGISTER_LAYER_NORM_PARAM_GRAD_GPU_KERNEL(dtype)                              \
  REGISTER_USER_KERNEL("layer_norm_param_grad")                                       \
      .SetCreateFn<LayerNormParamGradGpuKernel<dtype>>()                              \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                    \
        const user_op::TensorDesc* dy_desc = ctx.TensorDesc4ArgNameAndIndex("dy", 0); \
        return ctx.device_type() == DeviceType::kGPU                                  \
               && dy_desc->data_type() == GetDataType<dtype>::value;                  \
      });

REGISTER_LAYER_NORM_PARAM_GRAD_GPU_KERNEL(float)
REGISTER_LAYER_NORM_PARAM_GRAD_GPU_KERNEL(double)

}  // namespace oneflow
