#ifndef ONEFLOW_CORE_OPERATOR_DECODE_OFRECORD_OP_H_
#define ONEFLOW_CORE_OPERATOR_DECODE_OFRECORD_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/logical_node.h"

namespace oneflow {

class DecodeOFRecordOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DecodeOFRecordOp);
  DecodeOFRecordOp() = default;
  ~DecodeOFRecordOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  LogicalNode* NewProperLogicalNode() const override { return new DecodeLogicalNode; }

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx,
                             int64_t record_piece_size) const override;
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext* parallel_ctx,
                            KernelConf* kernel_conf) const override;

 private:
  Maybe<void> InferBatchAxis(
      std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const override;
  void GetSbpSignatures(SbpSignatureList* sbp_sig_list) const override;

  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_DECODE_OFRECORD_OP_H_
