#ifndef ONEFLOW_CORE_KERNEL_RELU_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RELU_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ReluKernelUtil final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReluKernelUtil);
  ReluKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const int64_t n, const T* in,
                      T* out);
  static void Backward(const KernelCtx& ctx, const int64_t n, const T* out_diff,
                       const T* in, T* in_diff);
};

template<DeviceType device_type, typename T>
class ReluKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReluKernel);
  ReluKernel() = default;
  ~ReluKernel() = default;

 private:
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_RELU_KERNEL_H_
