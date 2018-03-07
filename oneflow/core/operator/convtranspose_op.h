#ifndef ONEFLOW_CORE_OPERATOR_CONVTRANSPOSE_OP_H
#define ONEFLOW_CORE_OPERATOR_CONVTRANSPOSE_OP_H

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ConvtransposeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConvtransposeOp);
  ConvtransposeOp() = default;
  ~ConvtransposeOp() = default;

  void InitFromOpConf() override;

  bool NeedExtraInDiffMemWhenBackward() const override { return false; }
  bool NeedOutWhenBackward() const override { return false; }
  const PbMessage& GetSpecialConf() const override;
  void InferBlobDesc(
      std::function<BlobDesc*(const std::string)> GetBlobDesc5BnInOp,
      const ParallelContext* parallel_ctx) const override;

  int32_t ModelSplitAxis() const override { return 1; }
  int32_t MaxModeSplitNum() const override {
    return op_conf().convtranspose_conf().out_num();
  }

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONVTRANSPOSE_OP_H_
