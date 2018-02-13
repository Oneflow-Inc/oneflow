#ifndef ONEFLOW_CORE_KERNEL_AVERAGE_POOLING_3D_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_AVERAGE_POOLING_3D_KERNEL_H_

#include "oneflow/core/kernel/pooling_3d_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class AveragePooling3DKernel final : public Pooling3DKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePooling3DKernel);
  AveragePooling3DKernel() = default;
  ~AveragePooling3DKernel() = default;

 private:
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  const Pooling3DKernelConf& GetPooling3DKernelConf() const override;
  PoolingMode GetPoolingMode() override { return PoolingMode::kAveragePooling; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_AVERAGE_POOLING_3D_KERNEL_H_
