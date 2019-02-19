#ifndef ONEFLOW_CORE_OPERATOR_HALF_TO_FLOAT_OP_H_
#define ONEFLOW_CORE_OPERATOR_HALF_TO_FLOAT_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class HalfToFloatOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(HalfToFloatOp);
  HalfToFloatOp() = default;
  ~HalfToFloatOp() = default;

  void InitFromOpConf() override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().half_to_float_conf(); }
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return true; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_HALF_TO_FLOAT_OP_H_
