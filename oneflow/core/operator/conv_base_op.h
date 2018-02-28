#ifndef ONEFLOW_CORE_OPERATOR_CONV_BASE_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV_BASE_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

class CudnnConvNdDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnConvNdDesc);
  CudnnConvNdDesc() = delete;
  ~CudnnConvNdDesc();

  CudnnConvNdDesc(const BlobDesc*, const int, const std::vector<int>&,
                  const std::vector<int>&, const std::vector<int>&,
                  const std::string&, const std::string&);

  const cudnnConvolutionDescriptor_t& Get() const { return val_; }

 private:
  cudnnConvolutionDescriptor_t val_;
};

class ConvBaseOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvBaseOp);
  ConvBaseOp() = default;
  ~ConvBaseOp() = default;

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
  virtual PbMessage* MutableConvKernelConf(KernelConf* kernel_conf) const = 0;
  void SetCudnnConvAlgoForKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      KernelConf* kernel_conf) const;
  size_t InferCudnnWorkspaceSize(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp)
      const;

  const int kDimSize = -1;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV_BASE_OP_H_
