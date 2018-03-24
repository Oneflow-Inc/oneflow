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
class LocalResponseNormalizationKernel final : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LocalResponseNormalizationKernel);
  LocalResponseNormalizationKernel() = default;
  ~LocalResponseNormalizationKernel() = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override {
#ifdef WITH_CUDA
    const PbRf<int64_t>& shape = GetPbRfFromPbMessage<int64_t>(
        GetValFromPbMessage<const PbMessage&>(
            this->kernel_conf().local_response_normalization_conf(), "batch"),
        "dim");
    std::vector<int> dims(shape.begin(), shape.end());
    std::vector<int> strides{dims[1] * dims[2] * dims[3], 1, dims[2] * dims[3],
                             dims[3]};
    batch_desc_.reset(new CudnnTensorDesc(GetDataType<T>::val, 4, dims.data(),
                                          strides.data()));
    const LocalResponseNormalizationOpConf& op_conf =
        this->op_conf().local_response_normalization_conf();
    normalize_desc_.reset(new CudnnLRNDesc(op_conf.depth_radius(),
                                           op_conf.alpha(), op_conf.beta(),
                                           op_conf.bias()));
#endif  // WITH_CUDA
  }
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
#ifdef WITH_CUDA
  std::unique_ptr<CudnnTensorDesc> batch_desc_;
  std::unique_ptr<CudnnLRNDesc> normalize_desc_;
#endif  // WITH_CUDA
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_LOCAL_RESPONSE_NORMALIZATION_KERNEL_H_
