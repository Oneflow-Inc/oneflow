#ifndef ONEFLOW_OPERATOR_POOLING_OP_H_
#define ONEFLOW_OPERATOR_POOLING_OP_H_

#include <string>
#include "oneflow/operator/operator.h"

namespace oneflow {

class PoolingOp final : public UserOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PoolingOp);
  PoolingOp() = default;
  ~PoolingOp() = default;

  bool IsElemWise() const override { return true; }

  void InitFromOpConf(const OperatorConf& op_conf) override;
  const PbMessage& GetSpecialConf() const override;

  void InferShape4FwBlobs(
    std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
    ParallelPolicy policy,
    uint64_t parallel_id,
    uint64_t parallel_num) const override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_OPERATOR_POOLING_OP_H_
