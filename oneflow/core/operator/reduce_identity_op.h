#ifndef ONEFLOW_CORE_OPERATOR_REDUCE_IDENTITY_OP_H_
#define ONEFLOW_CORE_OPERATOR_REDUCE_IDENTITY_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class ReduceIdentityOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceIdentityOp);
  ReduceIdentityOp() = default;
  ~ReduceIdentityOp() = default;

  LogicalNode* NewProperLogicalNode() const override { return new ReduceIdentityLogicalNode; }
  void InitFromOpConf() override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().reduce_identity_conf(); }

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override {
    return NaiveInferBatchAxis(BatchAxis4BnInOp);
  }

  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_REDUCE_IDENTITY_OP_H_
