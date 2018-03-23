#ifndef ONEFLOW_CORE_OPERATOR_NORMALIZATION_OP_H_
#define ONEFLOW_CORE_OPERATOR_NORMALIZATION_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class NormalizationOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalizationOp);
  NormalizationOp() = default;
  ~NormalizationOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool IsElemWiseOp() const override { return true; }
  bool NeedExtraInDiffMemWhenBackward() const override { return false; }
  bool NeedOutWhenBackward() const override { return false; }
  bool IsNormalizationOp() const override { return true; }

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext* parallel_ctx) const override;

 private:
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext*, KernelConf*) const override;

  bool HasScaleOrCenter() const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_NORMALIZATION_OP_H_
