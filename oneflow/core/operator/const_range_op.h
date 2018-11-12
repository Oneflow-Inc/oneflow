#ifndef ONEFLOW_CORE_OPERATOR_CONST_RANGE_OP_H_
#define ONEFLOW_CORE_OPERATOR_CONST_RANGE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ConstRangeOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConstRangeOp);
  ConstRangeOp() = default;
  ~ConstRangeOp() override = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  bool IsAllOutputConst() const override { return true; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CONST_RANGE_OP_H_
