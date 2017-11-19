#ifndef ONEFLOW_CORE_KERNEL_RECORD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RECORD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class RecordKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecordKernel);
  RecordKernel() = default;
  ~RecordKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RECORD_KERNEL_H_
