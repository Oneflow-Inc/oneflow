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

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_NCCL_ALL_REDUCE_OP_H_
