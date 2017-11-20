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
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  using MemCopyFuncType = std::function<void(const KernelCtx& ctx, T*, T*,
                                             const int64_t, cudaMemcpyKind)>;

  void ConcatKernelWork(const KernelCtx&, const std::string&,
                        const std::vector<std::string>&,
                        std::function<Blob*(const std::string&)>,
                        MemCopyFuncType) const;

  void CopyDataIdToOb(const KernelCtx&, const std::vector<std::string>&,
                      const std::string&, const int32_t, cudaMemcpyKind,
                      std::function<Blob*(const std::string&)>) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CONCAT_KERNEL_H_
