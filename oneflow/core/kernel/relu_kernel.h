#ifndef ONEFLOW_CORE_KERNEL_RELU_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RELU_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
class ReluKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReluKernel);
  ReluKernel() = default;
  ~ReluKernel() = default;

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;
};

template<DeviceType device_type, typename FloatingPointType>
class ReluKernelUtil final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReluKernelUtil);
  ReluKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const int64_t n,
                      const FloatingPointType* in, FloatingPointType* out);
  static void Backward(const KernelCtx& ctx, const int64_t n,
                       const FloatingPointType* out_diff,
                       const FloatingPointType* in, FloatingPointType* in_diff);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RELU_KERNEL_H_
