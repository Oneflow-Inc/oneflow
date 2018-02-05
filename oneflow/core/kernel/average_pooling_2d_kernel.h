#ifndef ONEFLOW_CORE_KERNEL_AVERAGE_POOLING_2D_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_AVERAGE_POOLING_2D_KERNEL_H_

#include "oneflow/core/kernel/pooling_2d_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class AveragePooling2DKernel final : public Pooling2DKernel<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePooling2DKernel);
  AveragePooling2DKernel() = default;
  ~AveragePooling2DKernel() = default;

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

template<DeviceType device_type, typename T>
class AveragePooling2DKernelUtil {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AveragePooling2DKernelUtil);
  AveragePooling2DKernelUtil() = delete;

  static void Forward(const KernelCtx&, const Blob*, Blob*,
                      const Pooling2DCtx&);

  static void Backward(const KernelCtx&, const Blob*, Blob*,
                       const Pooling2DCtx&);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_AVERAGE_POOLING_2D_KERNEL_H_
