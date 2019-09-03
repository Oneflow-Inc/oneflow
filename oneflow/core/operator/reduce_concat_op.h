#ifndef ONEFLOW_CORE_OPERATOR_REDUCE_CONCAT_OP_H_
#define ONEFLOW_CORE_OPERATOR_REDUCE_CONCAT_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class ReduceConcatOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceConcatOp);
  ReduceConcatOp() = default;
  ~ReduceConcatOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  LogicalNode* NewProperLogicalNode() const override;
  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx, int64_t record_piece_size,
                             std::function<void(OpContext*)> EnrollOpCtx) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  Maybe<void> InferSbpSignature(
      SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
      const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
      std::function<const SbpInferHint&(const std::string&)> SbpInferHint4Ibn,
      const ParallelDesc& parallel_desc) const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*,
                            const OpContext* op_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_REDUCE_CONCAT_OP_H_
