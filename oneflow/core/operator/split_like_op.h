#ifndef ONEFLOW_CORE_OPERATOR_SPLIT_LIKE_OP_H
#define ONEFLOW_CORE_OPERATOR_SPLIT_LIKE_OP_H

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class SplitLikeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SplitLikeOp);
  SplitLikeOp() = default;
  ~SplitLikeOp() = default;

  void InitFromOpConf() override;

  const PbMessage& GetCustomizedConf() const override;

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  Maybe<void> InferBatchAxis(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;

  Maybe<void> GetSbpSignatures(
      const std::function<Maybe<const BlobDesc*>(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;

  int32_t FixAxis(const int32_t axis, const int64_t num_axes) const;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SPLIT_LIKE_OP_H
