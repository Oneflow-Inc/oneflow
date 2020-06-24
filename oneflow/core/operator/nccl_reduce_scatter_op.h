#ifndef ONEFLOW_CORE_OPERATOR_NCCL_REDUCE_SCATTER_OP_H_
#define ONEFLOW_CORE_OPERATOR_NCCL_REDUCE_SCATTER_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class NcclReduceScatterOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NcclReduceScatterOp);
  NcclReduceScatterOp() = default;
  ~NcclReduceScatterOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  Maybe<void> InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                             const ParallelContext* parallel_ctx) const override;

 private:
  LogicalBlobId lbi4ibn(const std::string& input_bn) const override { return GenPackedLbi(); }
  LogicalBlobId lbi4obn(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_NCCL_REDUCE_SCATTER_OP_H_
