#include <cstdint>
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<typename T>
class SmoothL1CPUKernel final : public user_op::OpKernel {
 public:
  SmoothL1CPUKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}

  SmoothL1CPUKernel() = default;
  ~SmoothL1CPUKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const float beta = ctx->GetAttr<float>("beta");
    const float scale = ctx->GetAttr<float>("scale");
    const user_op::Tensor* x_blob = ctx->Tensor4ArgNameAndIndex("x", 0);
    const T* x = x_blob->dptr<T>();
    const int64_t elem_cnt = x_blob->shape().elem_cnt();
    const T* label = ctx->Tensor4ArgNameAndIndex("label", 0)->dptr<T>();
    T* y = ctx->Tensor4ArgNameAndIndex("y", 0)->mut_dptr<T>();
    for (int64_t i = 0; i < elem_cnt; i++) {
      const T abs_diff = std::abs(x[i] - label[i]);
      if (abs_diff < beta) {
        y[i] = 0.5 * abs_diff * abs_diff / beta;
      } else {
        y[i] = abs_diff - 0.5 * beta;
      }
      y[i] *= scale;
    }
  };
};

#define REGISTER_SMOOTH_L1_CPU_KERNEL(dtype)                                                 \
  REGISTER_USER_KERNEL("smooth_l1")                                                          \
      .SetCreateFn(                                                                          \
          [](user_op::KernelInitContext* ctx) { return new SmoothL1CPUKernel<dtype>(ctx); }) \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                           \
        const user_op::TensorDesc* y_desc = ctx.TensorDesc4ArgNameAndIndex("y", 0);          \
        return ctx.device_type() == DeviceType::kCPU                                         \
               && y_desc->data_type() == GetDataType<dtype>::value;                          \
      });

REGISTER_SMOOTH_L1_CPU_KERNEL(float)
REGISTER_SMOOTH_L1_CPU_KERNEL(double)

template<typename T>
class SmoothL1GradCpuKernel final : public user_op::OpKernel {
 public:
  SmoothL1GradCpuKernel(user_op::KernelInitContext* ctx) : user_op::OpKernel(ctx) {}

  SmoothL1GradCpuKernel() = default;
  ~SmoothL1GradCpuKernel() = default;

 private:
  void Compute(user_op::KernelContext* ctx) override {
    const float beta = ctx->GetAttr<float>("beta");
    const float scale = ctx->GetAttr<float>("scale");
    const user_op::Tensor* x_blob = ctx->Tensor4ArgNameAndIndex("x", 0);
    const T* x = x_blob->dptr<T>();
    const int64_t elem_cnt = x_blob->shape().elem_cnt();
    const T* dy = ctx->Tensor4ArgNameAndIndex("dy", 0)->dptr<T>();
    const T* label = ctx->Tensor4ArgNameAndIndex("label", 0)->dptr<T>();
    T* dx = ctx->Tensor4ArgNameAndIndex("dx", 0)->mut_dptr<T>();
    for (int64_t i = 0; i < elem_cnt; i++) {
      const T diff = x[i] - label[i];
      const T abs_diff = std::abs(diff);
      if (abs_diff < beta) {
        dx[i] = abs_diff / beta;
      } else {
        dx[i] = (diff > GetZeroVal<T>()) - (diff < GetZeroVal<T>());
      }
      dx[i] *= scale * dy[i];
    }
  };
};

#define REGISTER_SMOOTH_L1_GRAD_CPU_KERNEL(dtype)                                                \
  REGISTER_USER_KERNEL("smooth_l1_grad")                                                         \
      .SetCreateFn(                                                                              \
          [](user_op::KernelInitContext* ctx) { return new SmoothL1GradCpuKernel<dtype>(ctx); }) \
      .SetIsMatchedPred([](const user_op::KernelRegContext& ctx) {                               \
        const user_op::TensorDesc* dx_desc = ctx.TensorDesc4ArgNameAndIndex("dx", 0);            \
        return ctx.device_type() == DeviceType::kCPU                                             \
               && dx_desc->data_type() == GetDataType<dtype>::value;                             \
      });

REGISTER_SMOOTH_L1_GRAD_CPU_KERNEL(float)
REGISTER_SMOOTH_L1_GRAD_CPU_KERNEL(double)

}  // namespace oneflow
