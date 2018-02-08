#ifndef ONEFLOW_CORE_KERNEL_MAX_POOLING_1D_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MAX_POOLING_1D_KERNEL_H_

#include "oneflow/core/kernel/pooling_1d_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class MaxPooling1DKernel final : public Pooling1DKernel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MaxPooling1DKernel);
  MaxPooling1DKernel() = default;
  ~MaxPooling1DKernel() = default;

 private:
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  const Pooling1DKernelConf& GetPooling1DKernelConf() const override;
  const PbMessage& GetPooling1DOpConf() const override;
  PoolingMode GetPoolingMode() override { return PoolingMode::kMaxPooling; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MAX_POOLING_1D_KERNEL_H_
