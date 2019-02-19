#ifndef ONEFLOW_CORE_KERNEL_SPARSE_CROSS_ENTROPY_LOSS_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_SPARSE_CROSS_ENTROPY_LOSS_KERNEL_H_

#include "oneflow/core/kernel/loss_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
class SparseCrossEntropyLossKernel final : public LossKernel<device_type, PredType> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SparseCrossEntropyLossKernel);
  SparseCrossEntropyLossKernel() = default;
  ~SparseCrossEntropyLossKernel() = default;

 private:
  void VirtualLossForwardDataContent(const KernelCtx&,
                                     std::function<Blob*(const std::string&)>) const override;
  const LossKernelConf& GetLossKernelConf(const KernelConf& kernel_conf) const override;
};

template<DeviceType device_type, typename PredType, typename LabelType>
struct SparseCrossEntropyLossKernelUtil {
  static void Forward(DeviceCtx* ctx, const int64_t instance_num, const int64_t num_of_classes,
                      const PredType* prediction, const LabelType* labels, PredType* loss);
  static void Backward(DeviceCtx* ctx, const int64_t instance_num, const int64_t num_of_classes,
                       const PredType* prediction, const LabelType* labels,
                       PredType* prediction_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_SPARSE_CROSS_ENTROPY_LOSS_KERNEL_H_
