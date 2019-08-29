#ifndef ONEFLOW_CORE_OPERATOR_SPARSE_CROSS_ENTROPY_GRAD_OP_H_
#define ONEFLOW_CORE_OPERATOR_SPARSE_CROSS_ENTROPY_GRAD_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class SparseCrossEntropyGradOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SparseCrossEntropyGradOp);
  SparseCrossEntropyGradOp() = default;
  ~SparseCrossEntropyGradOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, int64_t record_piece_size,
                             std::function<void(OpContext*)> EnrollOpCtx) const override;

 private:
  void GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override;
  Maybe<void> InferHasBatchDim(
      std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override {
    return NaiveInferHasBatchDim(HasBatchDim4BnInOp);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SPARSE_CROSS_ENTROPY_GRAD_OP_H_
