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

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  void InferHasBatchDim(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override {
    const SplitLikeOpConf& conf = op_conf().split_like_conf();
    int32_t split_axis = conf.axis();
    std::vector<int64_t> in_dim_vec = LogicalBlobDesc4Ibn("in").shape().dim_vec();
    if (split_axis < 0) { split_axis += in_dim_vec.size(); }
    bool has_batch_dim = true;
    if (split_axis == 0) { has_batch_dim = false; }
    for (const auto& obn : output_bns()) { *HasBatchDim4BnInOp(obn) = has_batch_dim; }
  }

  void GetSbpSignatures(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SPLIT_LIKE_OP_H
