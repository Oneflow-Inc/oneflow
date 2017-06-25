#ifndef ONEFLOW_CORE_KERNEL_INNERPRODUCT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_INNERPRODUCT_KERNEL_H_

#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename floating_point_type>
class InnerProductKernel;

template<typename floating_point_type>
class InnerProductKernel<DeviceType::kCPU, floating_point_type> final
    : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InnerProductKernel);
  InnerProductKernel() = default;
  ~InnerProductKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;
};

template<typename floating_point_type>
class InnerProductKernel<DeviceType::kGPU, floating_point_type> final
    : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InnerProductKernel);
  InnerProductKernel() = default;
  ~InnerProductKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_INNERPRODUCT_KERNEL_H_
