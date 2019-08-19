#ifndef ONEFLOW_CORE_OPERATOR_BATCH_GATHER_OP_H_
#define ONEFLOW_CORE_OPERATOR_BATCH_GATHER_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class BatchGatherOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BatchGatherOp);
  BatchGatherOp() = default;
  ~BatchGatherOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  void GetSbpSignatures(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
  void InferHasBatchDim(
      std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override {
    NaiveInferHasBatchDim(HasBatchDim4BnInOp);
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_BATCH_GATHER_OP_H_
