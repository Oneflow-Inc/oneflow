#ifndef ONEFLOW_CORE_KERNEL_CONV_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CONV_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/operator/conv_op.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ConvKernelIf : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvKernelIf);
  ConvKernelIf() = default;
  virtual ~ConvKernelIf() = default;

 protected:
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void InitPureModelTmpBlobs(
      DeviceCtx*,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelBlobsWithRandomSeed(
      DeviceCtx*, std::mt19937*,
      std::function<Blob*(const std::string&)>) const override;
  void InitModelBlobsWithDir(
      DeviceCtx*, int32_t part_id, int32_t part_num,
      const std::string& model_load_dir,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  virtual void VirtualWeightForward(DeviceCtx*, const Blob* in_blob,
                                    const Blob* weight_blob, Blob* out_blob,
                                    Blob* tmp) const {}
  virtual void VirtualBiasForward(DeviceCtx*, const Blob* bias_blob,
                                  Blob* out_blob) const {}
  virtual void VirtualDataBackward(DeviceCtx*, const Blob* out_diff_blob,
                                   const Blob* weight_blob, Blob* in_diff_blob,
                                   Blob* tmp) const {}
  virtual void VirtualWeightBackward(DeviceCtx*, const Blob* out_diff_blob,
                                     const Blob* in_blob,
                                     Blob* weight_diff_blob, Blob* tmp) const {}
  virtual void VirtualBiasBackward(DeviceCtx*, const Blob* out_diff_blob,
                                   Blob* bias_diff_blob) const {}

  const PbMessage& GetCustomizedOpConf() const override;
  const PbMessage& GetCustomizedKernelConf() const override;
  const int32_t KernelDim() const;
};

template<DeviceType device_type, typename T>
class ConvKernel;

template<typename T>
class ConvKernel<DeviceType::kCPU, T> final
    : public ConvKernelIf<DeviceType::kCPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvKernel);
  ConvKernel() = default;
  ~ConvKernel() = default;

 protected:
  void VirtualWeightForward(DeviceCtx*, const Blob* in_blob,
                            const Blob* weight_blob, Blob* out_blob,
                            Blob* tmp) const override;
  void VirtualBiasForward(DeviceCtx*, const Blob* bias_blob,
                          Blob* out_blob) const override;
  void VirtualDataBackward(DeviceCtx*, const Blob* out_diff_blob,
                           const Blob* weight_blob, Blob* in_diff_blob,
                           Blob* tmp) const override;
  void VirtualWeightBackward(DeviceCtx*, const Blob* out_diff_blob,
                             const Blob* in_blob, Blob* weight_diff_blob,
                             Blob* tmp) const override;
  void VirtualBiasBackward(DeviceCtx*, const Blob* out_diff_blob,
                           Blob* bias_diff_blob) const override;
};

template<typename T>
class ConvKernel<DeviceType::kGPU, T> final
    : public ConvKernelIf<DeviceType::kGPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvKernel);
  ConvKernel() = default;
  ~ConvKernel() = default;

 protected:
  void VirtualKernelInit(const ParallelContext*) override;
  void VirtualWeightForward(DeviceCtx*, const Blob* in_blob,
                            const Blob* weight_blob, Blob* out_blob,
                            Blob* tmp) const override;
  void VirtualBiasForward(DeviceCtx*, const Blob* bias_blob,
                          Blob* out_blob) const override;
  void VirtualDataBackward(DeviceCtx*, const Blob* out_diff_blob,
                           const Blob* weight_blob, Blob* in_diff_blob,
                           Blob* tmp) const override;
  void VirtualWeightBackward(DeviceCtx*, const Blob* out_diff_blob,
                             const Blob* in_blob, Blob* weight_diff_blob,
                             Blob* tmp) const override;
  void VirtualBiasBackward(DeviceCtx*, const Blob* out_diff_blob,
                           Blob* bias_diff_blob) const override;

 private:
  std::unique_ptr<CudnnTensorDesc> in_desc_;
  std::unique_ptr<CudnnTensorDesc> out_desc_;
  std::unique_ptr<CudnnFilterDesc> filter_desc_;
  std::unique_ptr<CudnnConvDesc> conv_desc_;
  std::unique_ptr<CudnnTensorDesc> bias_desc_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CONV_KERNEL_H_
