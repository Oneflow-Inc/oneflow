#ifndef ONEFLOW_CORE_OPERATOR_SCALAR_MUL_OP_H_
#define ONEFLOW_CORE_OPERATOR_SCALAR_MUL_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ScalarMulOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScalarMulOp);
  ScalarMulOp() = default;
  ~ScalarMulOp() = default;

  void InitFromOpConf() override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().scalar_mul_conf(); }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  void GetSbpSignatures(
      const std::function<const BlobDesc&(const std::string&)>& LogicalBlobDesc4Ibn,
      SbpSignatureList* sbp_sig_list) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SCALAR_MUL_OP_H_
