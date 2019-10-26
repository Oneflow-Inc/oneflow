#ifndef ONEFLOW_CORE_KERNEL_SPARSE_SOFTMAX_CROSS_ENTROPY_GRAD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SPARSE_SOFTMAX_CROSS_ENTROPY_GRAD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class SparseSoftmaxCrossEntropyGradKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SparseSoftmaxCrossEntropyGradKernel);
  SparseSoftmaxCrossEntropyGradKernel() = default;
  ~SparseSoftmaxCrossEntropyGradKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T, typename K>
struct SparseSoftmaxCrossEntropyGradKernelUtil {
  static void BackwardSub(DeviceCtx* ctx, const int64_t n, const int64_t w,
                          const int64_t lower_bound, const T* dy, const K* label, T* in_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SPARSE_SOFTMAX_CROSS_ENTROPY_GRAD_KERNEL_H_
