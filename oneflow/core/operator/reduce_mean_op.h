#ifndef ONEFLOW_CORE_OPERATOR_REDUCE_MEAN_OP_H_
#define ONEFLOW_CORE_OPERATOR_REDUCE_MEAN_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ReduceMeanOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceMeanOp);
  ReduceMeanOp() = default;
  ~ReduceMeanOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  void InferBwBufBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                           const ParallelContext*) const override;

 private:
  void InferHasBatchDim(std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override;

  void GetSbpSignatures(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override{};
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_REDUCE_MEAN_OP_H_
