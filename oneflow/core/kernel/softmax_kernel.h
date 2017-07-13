#ifndef ONEFLOW_CORE_KERNEL_SOFTMAX_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SOFTMAX_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
class SoftmaxKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxKernel);
  SoftmaxKernel() = default;
  ~SoftmaxKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename FloatingPointType>
class SoftmaxKernelUtil final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxKernelUtil);
  SoftmaxKernelUtil() = delete;

  // n = number of data sample
  // w = number of (input/output) neuron
  static void ForwardMax(const KernelCtx& ctx, const int64_t n, const int64_t w,
                         const FloatingPointType* out, FloatingPointType* tmp);

  static void ForwardSum(const KernelCtx& ctx, const int64_t n, const int64_t w,
                         const FloatingPointType* out, FloatingPointType* tmp);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SOFTMAX_KERNEL_H_
