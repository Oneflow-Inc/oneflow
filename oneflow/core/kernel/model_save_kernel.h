#ifndef ONEFLOW_CORE_KERNEL_MODEL_SAVE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MODEL_SAVE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<typename T>
class ModelSaveKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSaveKernel);
  ModelSaveKernel() = default;
  ~ModelSaveKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
};  // namespace oneflow

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MODEL_SAVE_KERNEL_H_
