#ifndef ONEFLOW_CORE_OPERATOR_NORMALIZATION_OP_H_
#define ONEFLOW_CORE_OPERATOR_NORMALIZATION_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

struct NormalizationOpCtx : public OpContext {
  int32_t axis;
  int32_t dims;
  int64_t transpose_rows;
  int64_t transpose_cols;
  bool need_transpose;
};

class NormalizationOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalizationOp);
  NormalizationOp() = default;
  ~NormalizationOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedExtraInDiffMemWhenBackward() const override { return false; }
  bool NeedOutWhenBackward() const override;

  void InferBlobDescs(
      std::function<BlobDesc*(const std::string)> GetBlobDesc4BnInOp,
      const ParallelContext*, DeviceType,
      std::function<void(OpContext*)> EnrollOpCtx) const override;

 private:
  void VirtualGenKernelConf(
      std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
      const ParallelContext*, KernelConf*, const OpContext*) const override;
  void VirtualFixParallelDesc(ParallelDesc* pr_desc) const override;
  NormalizationOpCtx* NewNormalizationOpCtx(const Shape& in_shape) const;

  bool HasScaleOrCenter() const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_NORMALIZATION_OP_H_
