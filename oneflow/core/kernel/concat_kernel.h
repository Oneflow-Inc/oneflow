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

  void Forward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void Backward(
      const KernelCtx& ctx,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

 private:
  using MemCopyFuncType = std::function<void(const KernelCtx& ctx, char*, char*,
                                             const int64_t, cudaMemcpyKind)>;

  void ConcatKernelWork(const KernelCtx&, const std::string&,
                        const PbRpf<std::string>&,
                        std::function<Blob*(const std::string&)>,
                        MemCopyFuncType) const;

  void CopyDataIdToOb(const KernelCtx&, const PbRpf<std::string>&,
                      const std::string&, const int32_t, cudaMemcpyKind,
                      std::function<Blob*(const std::string&)>) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CONCAT_KERNEL_H_
