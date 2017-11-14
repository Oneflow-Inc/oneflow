#ifndef ONEFLOW_CORE_KERNEL_RELU_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_RELU_KERNEL_H_

#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/kernel/kernel_manager.h"

namespace oneflow {

template<DeviceType device_type, typename FloatingPointType>
class ReluKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReluKernel);
  ReluKernel();
  ~ReluKernel();

  void Forward(const KernelCtx&,
               std::function<Blob*(const std::string&)>) const override;
  void Backward(const KernelCtx&,
                std::function<Blob*(const std::string&)>) const override;

 private:
#ifdef USE_CUDNN
  cudnnTensorDescriptor_t in_desc_;
  cudnnTensorDescriptor_t out_desc_;
  cudnnActivationDescriptor_t activ_desc_;
#endif  // USE_CUDNN
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
