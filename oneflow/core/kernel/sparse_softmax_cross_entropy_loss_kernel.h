#ifndef ONEFLOW_CORE_KERNEL_SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_KERNEL_H_

#include "oneflow/core/kernel/loss_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
class SparseSoftmaxCrossEntropyLossKernel final
    : public LossKernel<device_type, PredType, LabelType> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SparseSoftmaxCrossEntropyLossKernel);
  SparseSoftmaxCrossEntropyLossKernel() = default;
  ~SparseSoftmaxCrossEntropyLossKernel() = default;

 private:
  void VirtualLossForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  const LossKernelConf& GetLossKernelConf(
      const KernelConf& kernel_conf) const override;
};

template<DeviceType device_type, typename PredType, typename LabelType>
struct SparseSoftmaxCrossEntropyLossKernelUtil {
  static void BackwardSub(DeviceCtx* ctx, const int64_t n, const int64_t w,
                          const LabelType* label, PredType* in_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SPARSE_SOFTMAX_CROSS_ENTROPY_LOSS_KERNEL_H_
