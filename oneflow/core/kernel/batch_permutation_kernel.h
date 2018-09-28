#ifndef ONEFLOW_CORE_KERNEL_BATCH_PERMUTATION_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BATCH_PERMUTATION_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BatchPermutation final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BatchPermutation);
  BatchPermutation() = default;
  ~BatchPermutation() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
struct BatchPermutationUtil {
  static void Forward(const KernelCtx& ctx, const BatchPermutationOpConf& conf, const Blob* in_blob,
                      const Blob* indices_blob, Blob* out_blob);
  static void Backward(const KernelCtx& ctx, const BatchPermutationOpConf& conf,
                       const Blob* out_diff_blob, const Blob* indices_blob, Blob* in_diff_blob);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BATCH_PERMUTATION_KERNEL_H_
