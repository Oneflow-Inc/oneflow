#ifndef ONEFLOW_CORE_KERNEL_MODEL_SAVE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MODEL_SAVE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/job/job_conf.pb.h"

namespace oneflow {

template<DeviceType device_type, typename floating_point_type>
class ModelSaveKernel final {
};

template<typename floating_point_type>
class ModelSaveKernel<DeviceType::kCPU, floating_point_type> final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSaveKernel);
  ModelSaveKernel() = default;
  ~ModelSaveKernel() = default;

  void Forward(const KernelCtx&, std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx& kernel_ctx,
                std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const override {
    UNEXPECTED_RUN();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MODEL_SAVE_KERNEL_H_
