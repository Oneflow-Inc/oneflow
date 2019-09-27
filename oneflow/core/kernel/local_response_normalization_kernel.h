#ifndef ONEFLOW_CORE_KERNEL_LOCAL_RESPONSE_NORMALIZATION_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_LOCAL_RESPONSE_NORMALIZATION_KERNEL_H_

#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"

namespace oneflow {

#ifdef WITH_CUDA
class CudnnLRNDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnLRNDesc);
  CudnnLRNDesc() = delete;
  ~CudnnLRNDesc();

  CudnnLRNDesc(unsigned depth_radius, double alpha, double beta, double bias);

  const cudnnLRNDescriptor_t& Get() const { return val_; }

 private:
  cudnnLRNDescriptor_t val_;
};
#endif  // WITH_CUDA

template<DeviceType device_type, typename T>
class LocalResponseNormalizationKernel;

template<typename T>
class LocalResponseNormalizationKernel<DeviceType::kCPU, T> final
    : public KernelIf<DeviceType::kCPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalResponseNormalizationKernel);
  LocalResponseNormalizationKernel() = default;
  ~LocalResponseNormalizationKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void NCHWForward(const KernelCtx&, std::function<Blob*(const std::string&)>) const;
  void NHWCForward(const KernelCtx&, std::function<Blob*(const std::string&)>) const;
  void NCHWBackward(const KernelCtx&, std::function<Blob*(const std::string&)>) const;
  void NHWCBackward(const KernelCtx&, std::function<Blob*(const std::string&)>) const;
};

template<typename T>
class LocalResponseNormalizationKernel<DeviceType::kGPU, T> final
    : public KernelIf<DeviceType::kGPU> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalResponseNormalizationKernel);
  LocalResponseNormalizationKernel() = default;
  ~LocalResponseNormalizationKernel() = default;

 private:
  void VirtualKernelInit() override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
#ifdef WITH_CUDA
  std::unique_ptr<CudnnTensorDesc> batch_desc_;
  std::unique_ptr<CudnnLRNDesc> normalize_desc_;
#endif  // WITH_CUDA
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LOCAL_RESPONSE_NORMALIZATION_KERNEL_H_
