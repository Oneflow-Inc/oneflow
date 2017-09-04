#ifndef ONEFLOW_CORE_KERNEL_SOFTMAX_LOSS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SOFTMAX_LOSS_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
class SoftmaxLossKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxLossKernel);
  SoftmaxLossKernel() = default;
  ~SoftmaxLossKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override {
    UNEXPECTED_RUN();
  }
};

template<DeviceType device_type, typename PredType, typename LabelType>
class SoftmaxLossKernelUtil final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxLossKernelUtil);
  SoftmaxLossKernelUtil() = delete;

  static void ComputeLoss(DeviceCtx* ctx, const int64_t n, const int64_t w,
                          const LabelType* label, const PredType* prob,
                          PredType* tmp, PredType* loss);

  static void BackwardSub(DeviceCtx* ctx, const int64_t n, const int64_t w,
                          const LabelType* label, PredType* in_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SOFTMAX_LOSS_KERNEL_H_
