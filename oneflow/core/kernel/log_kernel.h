#ifndef ONEFLOW_CORE_KERNEL_LOG_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LOG_KERNEL_H_

#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
class LogKernel;

template<typename FloatingPointType>
class LogKernel<DeviceType::kCPU, FloatingPointType> final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LogKernel);
  LogKernel() = default;
  ~LogKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(
      const KernelCtx& kernel_ctx,
      std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const override {
    UNEXPECTED_RUN();
  }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LOG_KERNEL_H_
