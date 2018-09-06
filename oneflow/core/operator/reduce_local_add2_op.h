#ifndef ONEFLOW_CORE_OPERATOR_REDUCE_LOCAL_ADD_OP_H_
#define ONEFLOW_CORE_OPERATOR_REDUCE_LOCAL_ADD_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ReduceLocalAddOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceLocalAddOp);
  ReduceLocalAddOp() = default;
  ~ReduceLocalAddOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override { return GenPackedLbi(); }
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_REDUCE_LOCAL_ADD_OP_H_
