#ifndef ONEFLOW_CORE_KERNEL_REDUCE_SCATTER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_REDUCE_SCATTER_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

class ReduceScatterKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceScatterKernel);
  ReduceScatterKernel() = default;
  ~ReduceScatterKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_REDUCE_SCATTER_KERNEL_H_
