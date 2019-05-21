#ifndef ONEFLOW_CORE_KERNEL_BOXING_COPY_ADD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_BOXING_COPY_ADD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/register/tensor_partial_copier.h"
#include "oneflow/core/device/memory_copier.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class BoxingCopyAddKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingCopyAddKernel);
  BoxingCopyAddKernel() = default;
  ~BoxingCopyAddKernel() override = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void VirtualKernelInit(const ParallelContext*);

  std::vector<std::shared_ptr<TensorPartialCopier>> tensor_partial_copier_vec_;
  std::shared_ptr<MemoryCopier> memory_copier_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_BOXING_COPY_ADD_KERNEL_H_
