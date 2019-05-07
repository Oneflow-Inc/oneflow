#ifndef ONEFLOW_CORE_KERNEL_COPY_HD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_COPY_HD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

#ifdef WITH_CUDA

class CopyHdKernel final : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdKernel);
  CopyHdKernel() = default;
  ~CopyHdKernel() = default;

 private:
  bool HasSameShapeBetweenInOut() const override { return true; }
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  // Do calculate dynamic shape by call Blob::UpdateDynamicShapeIfNeed in copy_hd_actor
  // instead of implementing ForwardDim0ValidNum and ForwardInstanceShape here,
  // because all blobs fetched in CopyHdKernel are packed, for that dim0_valid_num and
  // instance_shape cannot be extracted.
};

#endif

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_COPY_HD_KERNEL_H_
