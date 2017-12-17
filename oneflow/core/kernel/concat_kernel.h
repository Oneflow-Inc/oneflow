#ifndef ONEFLOW_CORE_KERNEL_CONCAT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CONCAT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type>
class ConcatKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConcatKernel);
  ConcatKernel() = default;
  ~ConcatKernel() = default;

 private:
  void ConcatKernelWork(
      const KernelCtx& ctx, const std::string& obn,
      const PbRpf<std::string>& ibns,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const;

  void ForwardDataContent(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void ForwardDataId(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void BackwardDataContent(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CONCAT_KERNEL_H_
