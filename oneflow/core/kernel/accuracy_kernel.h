#ifndef ONEFLOW_CORE_KERNEL_ACCURACY_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_ACCURACY_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename PredType, typename LabelType>
class AccuracyKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccuracyKernel);
  AccuracyKernel() = default;
  ~AccuracyKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename PredType, typename LabelType>
struct AccuracyKernelUtil {
  static void Forward(DeviceCtx* ctx, const int32_t n, const int32_t d, int32_t top_k,
                      const PredType* prediction, const LabelType* label, PredType* accuracy);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ACCURACY_KERNEL_H_
