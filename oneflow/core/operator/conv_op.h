#ifndef ONEFLOW_CORE_OPERATOR_CONV_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

class CudnnConvDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnConvDesc);
  CudnnConvDesc() = delete;
  ~CudnnConvDesc();

  CudnnConvDesc(const BlobDesc*, const int, const std::vector<int>&,
                const std::vector<int>&, const std::vector<int>&,
                const std::string&, const std::string&);

  const cudnnConvolutionDescriptor_t& Get() const { return val_; }

 private:
  cudnnConvolutionDescriptor_t val_;
};

class ConvOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvOp);
  ConvOp() = default;
  ~ConvOp() = default;

  void InitFromOpConf() override;

  bool NeedExtraInDiffMemWhenBackward() const override { return false; }
  bool NeedOutWhenBackward() const override { return false; }
  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx,
      KernelConf* kernel_conf) const override;

 protected:
  PbMessage* MutableCustomizedKernelConf(
      KernelConf* kernel_conf) const override {
    return kernel_conf->mutable_conv_3d_conf();
  }
  virtual int32_t KernelDimSize() const = 0;
  void SetCudnnConvAlgoForKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      KernelConf* kernel_conf) const;
  size_t InferCudnnWorkspaceSize(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp)
      const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_OP_H_
