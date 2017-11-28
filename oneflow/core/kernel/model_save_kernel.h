#ifndef ONEFLOW_CORE_KERNEL_MODEL_SAVE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MODEL_SAVE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class ModelSaveKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSaveKernel);
  ModelSaveKernel() = default;
  ~ModelSaveKernel() = default;

 protected:
  void VirtualKernelInit(const ParallelContext*) override;
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void ForwardDataId(const KernelCtx&,
                     std::function<Blob*(const std::string&)>) const override {
    UNEXPECTED_RUN();
  }
  void BackwardDataId(const KernelCtx&,
                      std::function<Blob*(const std::string&)>) const override {
    UNEXPECTED_RUN();
  }

 private:
  int32_t part_id_;
  int32_t part_num_;
};  // namespace oneflow

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MODEL_SAVE_KERNEL_H_
