#ifndef ONEFLOW_CORE_OPERATOR_RESHAPE_OP_H_
#define ONEFLOW_CORE_OPERATOR_RESHAPE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ReshapeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReshapeOp);
  ReshapeOp() = default;
  ~ReshapeOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx,
                             const SbpSignature* sbp_signature) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      const ParallelDesc& parallel_desc, SbpSignatureList* sbp_sig_list) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_RESHAPE_OP_H_
