#ifndef ONEFLOW_CORE_OPERATOR_REDUCE_SUM_LIKE_OP_H_
#define ONEFLOW_CORE_OPERATOR_REDUCE_SUM_LIKE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ReduceSumLikeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceSumLikeOp);
  ReduceSumLikeOp() = default;
  ~ReduceSumLikeOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
  void InferHasBatchDim(std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override;
  void GetSbpSignatureRules(
      const std::function<const SbpInferHint&(const std::string&)>& SbpInferHint4Ibn,
      std::vector<std::unique_ptr<const SbpSignatureRule>>* rules) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_REDUCE_SUM_LIKE_OP_H_
