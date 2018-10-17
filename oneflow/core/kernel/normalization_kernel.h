#ifndef ONEFLOW_CORE_KERNEL_NORMALIZATION_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_NORMALIZATION_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/persistence/snapshot_manager.h"

namespace oneflow {

class NormalizationCtx final {
 public:
  NormalizationCtx(const KernelConf&, DataType);
  ~NormalizationCtx() = default;

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
class NormalizationKernel final : public KernelIfWithModel<device_type, T>,
                                  public KernelIfWithActivation<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalizationKernel);
  NormalizationKernel() = default;
  ~NormalizationKernel() = default;

 private:
  std::unique_ptr<NormalizationCtx> normalization_ctx_;
#ifdef WITH_CUDA
  void VirtualKernelInit(const ParallelContext*) override {
    if (this->kernel_conf().normalization_conf().use_cudnn()) {
      normalization_ctx_.reset(new NormalizationCtx(this->kernel_conf(), GetDataType<T>::value));
    }
  }
#endif  // WITH_CUDA
  void InitModelBlobsWithRandomSeed(
      DeviceCtx* ctx, std::mt19937* random_seed_gen,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelBlobsWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                             const std::string& model_load_dir,
                             std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)>) const override;

  void CalcAboutGammaDiff(const KernelCtx&, const std::function<Blob*(const std::string&)>,
                          const Blob* out_diff_blob, bool need_comp_in_diff) const;
  void CalcAboutBetaDiff(const KernelCtx&, const std::function<Blob*(const std::string&)>,
                         const Blob* out_diff_blob, bool need_comp_in_diff) const;
  void CalcInDiff(const KernelCtx&, const std::function<Blob*(const std::string&)>,
                  const Blob* out_diff_blob, Blob* in_diff_blob) const;
  void Normalize(const KernelCtx&, const std::function<Blob*(const std::string&)>&,
                 const Blob* mean_blob, const Blob* variance_blob, const Blob* in_blob,
                 Blob* out_blob) const;
  void CalcMeanAndVariance(const KernelCtx&, const std::function<Blob*(const std::string&)>&,
                           const Blob* in_blob) const;
  void UpdateMovingMeanAndMovingVariance(const KernelCtx&,
                                         const std::function<Blob*(const std::string&)>&) const;
  void InitMovingMeanAndMovingVariance(const KernelCtx& ctx,
                                       const std::function<Blob*(const std::string&)>& BnInOp2Blob,
                                       bool use_new) const;
  const PbMessage& GetCustomizedOpConf() const override;

  void NormalizationCudnnForward(const KernelCtx&,
                                 const std::function<Blob*(const std::string&)>&) const;
  void NormalizationCudnnBackward(const KernelCtx&,
                                  const std::function<Blob*(const std::string&)>&) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NORMALIZATION_KERNEL_H_
