#ifndef ONEFLOW_CORE_OPERATOR_REDUCE_CONCAT_OP_H_
#define ONEFLOW_CORE_OPERATOR_REDUCE_CONCAT_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ReduceConcatOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceConcatOp);
  ReduceConcatOp() = default;
  ~ReduceConcatOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx,
                      std::function<void(OpContext*)> EnrollOpCtx) const override;

 private:
  void VirtualGenKernelConf(std::function<const BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                            const ParallelContext*, KernelConf*,
                            const OpContext* op_ctx) const override;
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override { return GenPackedLbi(); }
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_REDUCE_CONCAT_OP_H_
