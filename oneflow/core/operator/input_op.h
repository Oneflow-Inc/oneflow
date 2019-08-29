#ifndef ONEFLOW_CORE_OPERATOR_INPUT_OP_H_
#define ONEFLOW_CORE_OPERATOR_INPUT_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class InputOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InputOp);
  InputOp() : Operator() {}
  ~InputOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx,
                             int64_t record_piece_size) const override;

 private:
  Maybe<void> InferHasBatchDim(
      std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override;
  void GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_INPUT_OP_H_
