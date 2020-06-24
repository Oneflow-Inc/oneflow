#include "oneflow/core/framework/framework.h"

namespace oneflow {

template<typename T>
class SmoothL1LossCPUKernel final : public user_op::OpKernel {
 public:
  SmoothL1LossCPUKernel() = default;
  ~SmoothL1LossCPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const float beta = ctx->Attr<float>("beta");
    const user_op::Tensor* prediction_blob = ctx->Tensor4ArgNameAndIndex("prediction", 0);
    const T* prediction = prediction_blob->dptr<T>();
    const int64_t elem_cnt = prediction_blob->shape().elem_cnt();
    const T* label = ctx->Tensor4ArgNameAndIndex("label", 0)->dptr<T>();
    T* loss = ctx->Tensor4ArgNameAndIndex("loss", 0)->mut_dptr<T>();
    for (int64_t i = 0; i < elem_cnt; i++) {
      const T abs_diff = std::abs(prediction[i] - label[i]);
      if (abs_diff < beta) {
        loss[i] = 0.5 * abs_diff * abs_diff / beta;
      } else {
        loss[i] = abs_diff - 0.5 * beta;
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SMOOTH_L1_LOSS_CPU_KERNEL(dtype)                   \
  REGISTER_USER_KERNEL("smooth_l1_loss")                            \
      .SetCreateFn<SmoothL1LossCPUKernel<dtype>>()                  \
      .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCPU \
                       & user_op::HobDataType("loss", 0) == GetDataType<dtype>::value);

REGISTER_SMOOTH_L1_LOSS_CPU_KERNEL(float)
REGISTER_SMOOTH_L1_LOSS_CPU_KERNEL(double)

template<typename T>
class SmoothL1LossGradCpuKernel final : public user_op::OpKernel {
 public:
  SmoothL1LossGradCpuKernel() = default;
  ~SmoothL1LossGradCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const float beta = ctx->Attr<float>("beta");
    const user_op::Tensor* prediction_blob = ctx->Tensor4ArgNameAndIndex("prediction", 0);
    const T* prediction = prediction_blob->dptr<T>();
    const int64_t elem_cnt = prediction_blob->shape().elem_cnt();
    const T* loss_grad = ctx->Tensor4ArgNameAndIndex("loss_grad", 0)->dptr<T>();
    const T* label = ctx->Tensor4ArgNameAndIndex("label", 0)->dptr<T>();
    T* prediction_grad = ctx->Tensor4ArgNameAndIndex("prediction_grad", 0)->mut_dptr<T>();
    for (int64_t i = 0; i < elem_cnt; i++) {
      const T diff = prediction[i] - label[i];
      const T abs_diff = std::abs(diff);
      if (abs_diff < beta) {
        prediction_grad[i] = diff / beta;
      } else {
        prediction_grad[i] = (diff > GetZeroVal<T>()) - (diff < GetZeroVal<T>());
      }
      prediction_grad[i] = prediction_grad[i] * loss_grad[i];
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SMOOTH_L1_LOSS_GRAD_CPU_KERNEL(dtype)              \
  REGISTER_USER_KERNEL("smooth_l1_loss_grad")                       \
      .SetCreateFn<SmoothL1LossGradCpuKernel<dtype>>()              \
      .SetIsMatchedHob(user_op::HobDeviceType() == DeviceType::kCPU \
                       & user_op::HobDataType("prediction_grad", 0) == GetDataType<dtype>::value);

REGISTER_SMOOTH_L1_LOSS_GRAD_CPU_KERNEL(float)
REGISTER_SMOOTH_L1_LOSS_GRAD_CPU_KERNEL(double)

}  // namespace oneflow
