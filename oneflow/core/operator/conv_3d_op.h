#ifndef ONEFLOW_CORE_OPERATOR_CONV3D_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONV3D_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

class CudnnConvNdDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnConvNdDesc);
  CudnnConvNdDesc() = delete;
  ~CudnnConvNdDesc();

  CudnnConvNdDesc(std::function<const BlobDesc*(const std::string)>,
                  const Conv3dOpConf&);

  const cudnnConvolutionDescriptor_t& Get() const { return val_; }

 private:
  cudnnConvolutionDescriptor_t val_;
};

class Conv3dOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Conv3dOp);
  Conv3dOp() = default;
  ~Conv3dOp() = default;

  void InitFromOpConf() override;

  bool NeedExtraInDiffMemWhenBackward() const override { return false; }
  bool NeedOutWhenBackward() const override { return false; }
  const PbMessage& GetSpecialConf() const override;
  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

  int32_t ModelSplitAxis() const override { return 1; }
  int32_t MaxModelSplitNum() const override {
    return op_conf().conv_3d_conf().filters();
  }
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx,
      KernelConf* kernel_conf) const override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONV3D_OP_H_
