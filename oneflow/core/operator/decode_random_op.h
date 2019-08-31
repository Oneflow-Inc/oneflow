#ifndef ONEFLOW_CORE_OPERATOR_DECODE_RANDOM_OP_H_
#define ONEFLOW_CORE_OPERATOR_DECODE_RANDOM_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class DecodeRandomOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeRandomOp);
  DecodeRandomOp() = default;
  ~DecodeRandomOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  LogicalNode* NewProperLogicalNode() const override { return new DecodeRandomLogicalNode; }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx,
                             int64_t record_piece_size) const override;

 private:
  Maybe<void> InferHasBatchDim(
      std::function<bool*(const std::string&)> HasBatchDim4BnInOp) const override;
  void GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_DECODE_RANDOM_OP_H_
