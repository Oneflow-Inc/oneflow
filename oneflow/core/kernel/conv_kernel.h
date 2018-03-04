#ifndef ONEFLOW_CORE_KERNEL_CONV_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_CONV_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/operator/conv_op.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class ConvKernel : public KernelIf<device_type> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvKernel);
  ConvKernel() = default;
  virtual ~ConvKernel() = default;

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

  const PbMessage& GetCustomizedOpConf() const override {
    CHECK(this->kernel_conf().has_conv_conf());
    switch (this->GetInt32FromCustomizedKernelConf("kernel_dim_size")) {
      case 1: return this->op_conf().conv_1d_conf();
      case 2: return this->op_conf().conv_2d_conf();
      case 3: return this->op_conf().conv_3d_conf();
      default: UNIMPLEMENTED();
    }
  }
  const PbMessage& GetCustomizedKernelConf() const override {
    return this->kernel_conf().conv_conf();
  }
};

#ifdef WITH_CUDA
template<typename T>
class CudnnConvKernel final : public ConvKernel<DeviceType::kGPU, T> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnConvKernel);
  CudnnConvKernel() = default;
  ~CudnnConvKernel() = default;

 protected:
  void VirtualKernelInit(const ParallelContext*) override;
  void ForwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;
  void BackwardDataContent(
      const KernelCtx&,
      std::function<Blob*(const std::string&)>) const override;

 private:
  std::unique_ptr<CudnnTensorDesc> in_desc_;
  std::unique_ptr<CudnnTensorDesc> out_desc_;
  std::unique_ptr<CudnnFilterDesc> filter_desc_;
  std::unique_ptr<CudnnConvDesc> conv_desc_;
  std::unique_ptr<CudnnTensorDesc> bias_desc_;
};
#endif  // WITH_CUDA

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_CONV_KERNEL_H_
