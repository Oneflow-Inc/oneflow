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
      const T abs_x = std::abs(x[i] - label[i]);
      if (abs_x < beta) {
        y[i] = 0.5 * abs_x * abs_x / beta;
      } else {
        y[i] = abs_x - 0.5 * beta;
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

}  // namespace oneflow
