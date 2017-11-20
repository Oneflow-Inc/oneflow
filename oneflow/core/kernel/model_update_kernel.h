#ifndef ONEFLOW_CORE_KERNEL_MODEL_UPDATE_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_MODEL_UPDATE_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class ModelUpdtKernel : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelUpdtKernel);
  virtual ~ModelUpdtKernel() = default;

  virtual void InitDataTmpBlobs(
      const KernelCtx&, std::function<Blob*(const std::string&)>) const {
    UNEXPECTED_RUN();
  }

 protected:
  ModelUpdtKernel() = default;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_MODEL_UPDATE_KERNEL_H_
