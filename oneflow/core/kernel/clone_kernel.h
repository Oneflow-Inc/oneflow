#ifndef ONEFLOW_CORE_KERNEL_CLONE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CLONE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/job/job_conf.pb.h"

namespace oneflow {

template<DeviceType device_type, typename floating_point_type>
class CloneKernel final {
};

template<typename floating_point_type>
class CloneKernel<DeviceType::kCPU, floating_point_type> final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CloneKernel);
  CloneKernel() = default;
  ~CloneKernel() = default;

  void Forward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;
};

template<typename floating_point_type>
class CloneKernel<DeviceType::kGPU, floating_point_type> final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CloneKernel);
  CloneKernel() = default;
  ~CloneKernel() = default;

  void Forward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_KERNEL_CLONE_KERNEL_H_
