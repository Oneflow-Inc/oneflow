#ifndef ONEFLOW_CORE_OPERATOR_AFFINE_CHANNEL_OP_H_
#define ONEFLOW_CORE_OPERATOR_AFFINE_CHANNEL_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/normalization_op.h"
#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

class AffineChannelOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AffineChannelOp);
  AffineChannelOp() = default;
  ~AffineChannelOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedOutBlobWhenBackward() const override { return false; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext*,
                      std::function<void(OpContext*)> EnrollOpCtx) const override;

 private:
  void InferParamBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                           const AffineChannelOpConf&, int64_t norm_part_num, DataType in_data_type,
                           bool use_cudnn) const;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*, const OpContext*) const override;
#ifdef WITH_CUDA
  void InferBlobDescsForCudnn(
      std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp) const;
  void VirtualGenKernelConfForCudnn(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp, const ParallelContext*,
      KernelConf*) const;
#endif
  void VirtualFixParallelDesc(ParallelDesc* pr_desc) const override;
  NormalizationOpCtx* NewNormalizationOpCtx(const Shape& in_shape) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_AFFINE_CHANNEL_OP_H_
