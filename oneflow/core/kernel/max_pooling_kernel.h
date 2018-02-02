#ifndef ONEFLOW_CORE_KERNEL_MAX_POOLING_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MAX_POOLING_KERNEL_H_

#include "oneflow/core/kernel/pooling_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MaxPoolingKernel final : public PoolingKernel<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPoolingKernel);
  MaxPoolingKernel() = default;
  ~MaxPoolingKernel() = default;

 private:
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  const PoolingKernelConf& GetPoolingKernelConf() const override;
};

template<DeviceType device_type, typename T>
class MaxPoolingKernelUtil {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPoolingKernelUtil);
  MaxPoolingKernelUtil() = delete;

  static void Forward(const KernelCtx&, const Blob*, Blob*, Blob*,
                      const PoolingCtx&);

  static void Backward(const KernelCtx&, const Blob*, const Blob*, Blob*,
                       const PoolingCtx&);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MAX_POOLING_KERNEL_H_
