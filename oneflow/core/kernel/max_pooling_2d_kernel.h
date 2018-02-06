#ifndef ONEFLOW_CORE_KERNEL_MAX_POOLING_2D_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MAX_POOLING_2D_KERNEL_H_

#include "oneflow/core/kernel/pooling_2d_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MaxPooling2DKernel final : public Pooling2DKernel<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPooling2DKernel);
  MaxPooling2DKernel() = default;
  ~MaxPooling2DKernel() = default;

 private:
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  const Pooling2DKernelConf& GetPooling2DKernelConf() const override;
  const PbMessage& GetPooling2DOpConf() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MAX_POOLING_2D_KERNEL_H_
