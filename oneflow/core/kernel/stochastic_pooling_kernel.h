#ifndef ONEFLOW_CORE_KERNEL_STOCHASTIC_POOLING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_STOCHASTIC_POOLING_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class StochasticPoolingKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StochasticPoolingKernel);
  StochasticPoolingKernel() = default;
  ~StochasticPoolingKernel() = default;

 private:
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
class StochasticPoolingKernelUtil {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StochasticPoolingKernelUtil);
  StochasticPoolingKernelUtil() = delete;

  static void PoolingForward(const KernelCtx&, const Blob*, Blob*, Blob*,
                             const StochasticPoolingOpConf&);

  static void PoolingBackward(const KernelCtx&, const Blob*, const Blob*, Blob*,
                              const StochasticPoolingOpConf&);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_STOCHASTIC_POOLING_KERNEL_H_
