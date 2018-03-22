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
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitPureModelTmpBlobs(
      DeviceCtx*,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelBlobsWithRandomSeed(
      DeviceCtx*, std::mt19937* random_seed_gen,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void InitModelBlobsWithDir(
      DeviceCtx*, int32_t part_id, int32_t part_num,
      const std::string& model_load_dir,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  virtual void WeightForward(
      DeviceCtx*, const Blob* in, const Blob* weight, Blob* out,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;
  virtual void BiasForward(DeviceCtx*, const Blob* bias, Blob* out) const = 0;
  virtual void DataBackward(
      DeviceCtx*, const Blob* out_diff, const Blob* weight, Blob* in_diff,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;
  virtual void WeightBackward(
      DeviceCtx*, const Blob* out_diff, const Blob* in, Blob* weight_diff,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const = 0;
  virtual void BiasBackward(DeviceCtx*, const Blob* out_diff,
                            Blob* bias_diff) const = 0;

  const PbMessage& GetCustomizedOpConf() const override;
  const ConvKernelConf& GetConvKernelConf() const;
  const int32_t OpKernelDim() const;
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

 private:
  void WeightForward(
      DeviceCtx*, const Blob* in, const Blob* weight, Blob* out,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BiasForward(DeviceCtx*, const Blob* bias, Blob* out) const override;
  void DataBackward(
      DeviceCtx*, const Blob* out_diff, const Blob* weight, Blob* in_diff,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void WeightBackward(
      DeviceCtx*, const Blob* out_diff, const Blob* in, Blob* weight_diff,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BiasBackward(DeviceCtx*, const Blob* out_diff,
                    Blob* bias_diff) const override;
};

template<typename T>
class ConvKernel<DeviceType::kGPU, T> final
    : public ConvKernelIf<DeviceType::kGPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvKernel);
  ConvKernel() = default;
  ~ConvKernel() = default;

 private:
  void VirtualKernelInit(const ParallelContext*) override;
  void WeightForward(
      DeviceCtx*, const Blob* in, const Blob* weight, Blob* out,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BiasForward(DeviceCtx*, const Blob* bias, Blob* out) const override;
  void DataBackward(
      DeviceCtx*, const Blob* out_diff, const Blob* weight, Blob* in_diff,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void WeightBackward(
      DeviceCtx*, const Blob* out_diff, const Blob* in, Blob* weight_diff,
      std::function<Blob*(const std::string&)> BnInOp2Blob) const override;
  void BiasBackward(DeviceCtx*, const Blob* out_diff,
                    Blob* bias_diff) const override;

  std::unique_ptr<CudnnTensorDesc> in_desc_;
  std::unique_ptr<CudnnTensorDesc> out_desc_;
  std::unique_ptr<CudnnFilterDesc> filter_desc_;
  std::unique_ptr<CudnnConvDesc> conv_desc_;
  std::unique_ptr<CudnnTensorDesc> bias_desc_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CONV_KERNEL_H_
