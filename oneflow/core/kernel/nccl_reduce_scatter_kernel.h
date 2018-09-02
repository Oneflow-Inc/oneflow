#ifndef ONEFLOW_CORE_KERNEL_NCCL_REDUCE_SCATTER_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_NCCL_REDUCE_SCATTER_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class NcclReduceScatterKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclReduceScatterKernel);
  NcclReduceScatterKernel() = default;
  ~NcclReduceScatterKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NCCL_REDUCE_SCATTER_KERNEL_H_
