#ifndef ONEFLOW_CORE_KERNEL_COPY_HD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_COPY_HD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename floating_point_type>
class CopyHdKernel final {
};

template<typename floating_point_type>
class CopyHdKernel<DeviceType::kCPU, floating_point_type> final
    : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdKernel);
  CopyHdKernel() = default;
  ~CopyHdKernel() = default;

  void Forward(const KernelContext&,
               std::function<Blob*(const std::string&)>) const override {}
  void Backward(const KernelContext&,
                std::function<Blob*(const std::string&)>) const override {}
};

template<typename floating_point_type>
class CopyHdKernel<DeviceType::kGPU, floating_point_type> final 
    : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdKernel);
  CopyHdKernel() = default;
  ~CopyHdKernel() = default;

  void Forward(const KernelContext&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelContext&,
                std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_COPY_HD_KERNEL_H_
