#ifndef ONEFLOW_CORE_KERNEL_CONCAT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CONCAT_KERNEL_H_

#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ConcatKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConcatKernel);
  ConcatKernel() = default;
  ~ConcatKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;

 private:
  using MemCopyFuncType = std::function<void(const KernelCtx& ctx, T*, T*,
                                             const int64_t, cudaMemcpyKind)>;

  void ConcatKernelWork(const KernelCtx&, const std::string&,
                        const std::vector<std::string>&,
                        std::function<Blob*(const std::string&)>,
                        MemCopyFuncType) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CONCAT_KERNEL_H_
