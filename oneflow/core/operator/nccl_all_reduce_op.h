#ifndef ONEFLOW_CORE_OPERATOR_NCCL_ALL_REDUCE_OP_H_
#define ONEFLOW_CORE_OPERATOR_NCCL_ALL_REDUCE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class NcclAllReduceOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclAllReduceOp);
  NcclAllReduceOp() = default;
  ~NcclAllReduceOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { UNIMPLEMENTED(); }
  LogicalNode* NewProperLogicalNode() override;
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_NCCL_ALL_REDUCE_OP_H_
