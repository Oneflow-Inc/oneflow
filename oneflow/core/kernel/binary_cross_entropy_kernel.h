#ifndef ONEFLOW_CORE_KERNEL_BINARY_CROSS_ENTROPY_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BINARY_CROSS_ENTROPY_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BinaryCrossEntropyKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BinaryCrossEntropyKernel);
  BinaryCrossEntropyKernel() = default;
  ~BinaryCrossEntropyKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T, typename K>
struct BinaryCrossEntropyKernelUtil {
  static void ComputeEntropy(DeviceCtx* ctx, int64_t num_instances, const T* x, const K* labels,
                             T* y);
  static void ComputeDiff(DeviceCtx* ctx, int64_t num_instances, const T* x, const K* labels,
                          const T* dy, T* dx);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BINARY_CROSS_ENTROPY_KERNEL_H_
