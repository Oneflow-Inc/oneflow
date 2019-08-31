#ifndef ONEFLOW_CORE_OPERATOR_NCCL_ALL_GATHER_OP_H_
#define ONEFLOW_CORE_OPERATOR_NCCL_ALL_GATHER_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class NcclAllGatherOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclAllGatherOp);
  NcclAllGatherOp() = default;
  ~NcclAllGatherOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override { return GenPackedLbi(); }
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_NCCL_ALL_GATHER_OP_H_
