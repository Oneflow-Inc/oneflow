#ifndef ONEFLOW_CORE_KERNEL_AVERAGE_POOLING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_AVERAGE_POOLING_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class AveragePoolingKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePoolingKernel);
  AveragePoolingKernel() = default;
  ~AveragePoolingKernel() = default;

 private:
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename T>
class AveragePoolingKernelUtil {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePoolingKernelUtil);
  AveragePoolingKernelUtil() = delete;

  static void PoolingForward(const KernelCtx&, const Blob*, Blob*,
                             const AveragePoolingOpConf&);

  static void PoolingBackward(const KernelCtx&, const Blob*, Blob*,
                              const AveragePoolingOpConf&);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_AVERAGE_POOLING_KERNEL_H_
