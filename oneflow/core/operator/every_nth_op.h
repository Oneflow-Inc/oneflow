#ifndef ONEFLOW_CORE_OPERATOR_EVERY_NTH_OP_H_
#define ONEFLOW_CORE_OPERATOR_EVERY_NTH_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class EveryNthOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EveryNthOp);
  EveryNthOp() = default;
  ~EveryNthOp() override = default;

 private:
  Maybe<void> InferHasBatchDim(
      std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override {
    return NaiveInferHasBatchDim(HasBatchDim4BnInOp);
  }
  void GetSbpSignatures(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  Maybe<void> InferOutputBlobTimeShape(
      std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
      const ParallelContext* parallel_ctx, Shape* time_shape) const override;
  void InitFromOpConf() override;
  LogicalNode* NewProperLogicalNode() const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_EVERY_NTH_OP_H_
