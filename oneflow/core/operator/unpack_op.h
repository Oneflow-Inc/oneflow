#ifndef ONEFLOW_CORE_OPERATOR_UNPACK_OP_H_
#define ONEFLOW_CORE_OPERATOR_UNPACK_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class UnpackOp final : public Operator {
 public:
  OF_DISALLOW_COPY(UnpackOp);
  UnpackOp() = default;
  ~UnpackOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().unpack_conf(); }
  LogicalNode* NewProperLogicalNode() const override { return new UnpackForwardLogicalNode; }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
  Maybe<void> InferOutputBlobTimeShape(
      std::function<const Shape*(const std::string&)> GetTimeShape4BnInOp,
      const ParallelContext* parallel_ctx, Shape* time_shape) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_UNPACK_OP_H_
