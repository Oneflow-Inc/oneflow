#ifndef ONEFLOW_CORE_KERNEL_CONCAT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CONCAT_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ConcatKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConcatKernel);
  ConcatKernel() = default;
  ~ConcatKernel() = default;

 private:
  void ConcatKernelWork(const KernelCtx& ctx, const std::string& obn,
                        const PbRpf<std::string>& ibns,
                        std::function<Blob*(const std::string&)> BnInOp2Blob) const;

  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void ForwardDataId(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void ForwardColNum(const KernelCtx& ctx,
                     std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void BackwardDataContent(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  void ForwardField(const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob,
                    std::function<char*(Blob*)> GetOutBlobField,
                    std::function<const char*(Blob*)> GetInBlobField,
                    std::function<size_t(Blob*)> GetFieldSize) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CONCAT_KERNEL_H_
