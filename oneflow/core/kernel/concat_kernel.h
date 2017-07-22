#ifndef ONEFLOW_CORE_KERNEL_CONCAT_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CONCAT_KERNEL_H_

#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
class ConcatKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConcatKernel);
  ConcatKernel() = default;
  ~ConcatKernel() = default;

  using DualCopy =
      std::function<void(const KernelCtx& ctx, FloatingPointType*,
                         FloatingPointType*, const int64_t, cudaMemcpyKind)>;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;
  void ForOrBackWard(const KernelCtx&, const std::string,
                     const std::vector<std::string>,
                     std::function<Blob*(const std::string&)>, DualCopy) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CONCAT_KERNEL_H_
