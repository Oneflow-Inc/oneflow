#ifndef ONEFLOW_CORE_KERNEL_AFFINE_CHANNEL_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_AFFINE_CHANNEL_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/normalization_kernel.h"
#include "oneflow/core/persistence/snapshot_manager.h"

namespace oneflow {

class AffineChannelCtx final {
 public:
  AffineChannelCtx(const KernelConf&, DataType);
  ~AffineChannelCtx() = default;

#ifdef WITH_CUDA
  const cudnnBatchNormMode_t& cudnn_batch_norm_mode() const;
  const cudnnTensorDescriptor_t& cudnn_in_tensor_desc() const;
  const cudnnTensorDescriptor_t& cudnn_param_tensor_desc() const;

 private:
  cudnnBatchNormMode_t mode_;
  std::unique_ptr<CudnnTensorDesc> in_desc_;
  std::unique_ptr<CudnnTensorDesc> param_desc_;
#endif  // WITH_CUDA
};

template<DeviceType device_type, typename T>
class AffineChannelKernel final : public KernelIfWithModel<device_type, T>,
                                  public KernelIfWithActivation<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AffineChannelKernel);
  AffineChannelKernel() = default;
  ~AffineChannelKernel() = default;

 private:
  std::unique_ptr<AffineChannelCtx> affine_channel_ctx_;
#ifdef WITH_CUDA
  void VirtualKernelInit(const ParallelContext*) override {
    if (this->kernel_conf().affine_channel_conf().use_cudnn()) {
      affine_channel_ctx_.reset(new AffineChannelCtx(this->kernel_conf(), GetDataType<T>::value));
    }
  }
#endif  // WITH_CUDA
  void InitModelBlobsWithRandomSeed(
      DeviceCtx* ctx, std::mt19937* random_seed_gen,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelBlobsWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                             const std::string& model_load_dir,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitConstBufBlobs(DeviceCtx* ctx,
                         std::function<Blob*(const std::string&)> BnInOp2Blob) const;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;
  const PbMessage& GetCustomizedOpConf() const override;

  void AffineChannelCudnnForward(const KernelCtx&,
                                 const std::function<Blob*(const std::string&)>&) const;
  void AffineChannelCudnnBackward(const KernelCtx&,
                                  const std::function<Blob*(const std::string&)>&) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_AFFINE_CHANNEL_KERNEL_H_
