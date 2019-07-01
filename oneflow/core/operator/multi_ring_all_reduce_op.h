#ifndef ONEFLOW_CORE_OPERATOR_MULTI_RING_ALL_REDUCE_OP_H_
#define ONEFLOW_CORE_OPERATOR_MULTI_RING_ALL_REDUCE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class MultiRingAllReduceOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultiRingAllReduceOp);
  MultiRingAllReduceOp() = default;
  ~MultiRingAllReduceOp() override = default;

  void InitFromOpConf() override;

 private:
  LogicalBlobId ibn2lbi(const std::string& input_bn) const override;
  LogicalBlobId obn2lbi(const std::string& output_bn) const override;
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // #define ONEFLOW_CORE_OPERATOR_MULTI_RING_ALL_REDUCE_OP_H_
