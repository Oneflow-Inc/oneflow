#ifndef ONEFLOW_CORE_OPERATOR_FULLY_CONNECTED_OP_H_
#define ONEFLOW_CORE_OPERATOR_FULLY_CONNECTED_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class FullyConnectedOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FullyConnectedOp);
  FullyConnectedOp() = default;
  ~FullyConnectedOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> InferHasBatchDim(
      std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override {
    return NaiveInferHasBatchDim(HasBatchDim4BnInOp);
  }

  void GetSbpSignatures(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_FULLY_CONNECTED_OP_H_
