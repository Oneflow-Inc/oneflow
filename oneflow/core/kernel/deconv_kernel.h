#ifndef ONEFLOW_CORE_KERNEL_DECONV_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_DECONV_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/operator/deconv_op.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class DeconvKernelIf : public KernelIfWithActivation<device_type, T>,
                       public KernelIfWithModel<device_type, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeconvKernelIf);
  DeconvKernelIf() = default;
  virtual ~DeconvKernelIf() = default;

 protected:
  void ForwardDataContent(const KernelCtx&,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BackwardDataContent(const KernelCtx&,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  virtual void DoForwardDataContent(DeviceCtx*, const Blob* in_blob, const Blob* weight_blob,
                                    Blob* out_blob,
                                    std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;
  virtual void WeightBackward(DeviceCtx*, const Blob* out_diff_blob, const Blob* in_blob,
                              Blob* weight_diff_blob, Blob* in_diff_blob,
                              std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;
  virtual void BiasBackward(DeviceCtx*, const Blob* out_diff_blob, Blob* bias_diff_blob,
                            std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;

  const PbMessage& GetCustomizedOpConf() const override;
  const DeconvKernelConf& GetDeconvKernelConf() const;
  const int32_t OpKernelDim() const;
};

template<DeviceType device_type, typename T>
class DeconvKernel;

template<typename T>
class DeconvKernel<DeviceType::kCPU, T> final : public DeconvKernelIf<DeviceType::kCPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeconvKernel);
  DeconvKernel() = default;
  ~DeconvKernel() = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  void DoForwardDataContent(DeviceCtx*, const Blob* in_blob, const Blob* weight_blob,
                            Blob* out_blob,
                            std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void WeightBackward(DeviceCtx*, const Blob* out_diff_blob, const Blob* in_blob,
                      Blob* weight_diff_blob, Blob* in_diff_blob,
                      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BiasBackward(DeviceCtx*, const Blob* out_diff_blob, Blob* bias_diff_blob,
                    std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
};

template<typename T>
class DeconvKernel<DeviceType::kGPU, T> final : public DeconvKernelIf<DeviceType::kGPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DeconvKernel);
  DeconvKernel() = default;
  ~DeconvKernel() = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  void DoForwardDataContent(DeviceCtx*, const Blob* in_blob, const Blob* weight_blob,
                            Blob* out_blob,
                            std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void WeightBackward(DeviceCtx*, const Blob* out_diff_blob, const Blob* in_blob,
                      Blob* weight_diff_blob, Blob* in_diff_blob,
                      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BiasBackward(DeviceCtx*, const Blob* out_diff_blob, Blob* bias_diff_blob,
                    std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  std::unique_ptr<CudnnTensorDesc> in_desc_;
  std::unique_ptr<CudnnTensorDesc> out_desc_;
  std::unique_ptr<CudnnFilterDesc> filter_desc_;
  std::unique_ptr<CudnnDeconvDesc> deconv_desc_;
  std::unique_ptr<CudnnTensorDesc> bias_desc_;
};

}  //  namespace oneflow

#endif  //  ONEFLOW_CORE_KERNEL_DECONV_KERNEL_H
