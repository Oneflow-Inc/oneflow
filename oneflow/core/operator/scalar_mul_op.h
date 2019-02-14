#ifndef ONEFLOW_CORE_OPERATOR_SCALAR_MUL_OP_H_
#define ONEFLOW_CORE_OPERATOR_SCALAR_MUL_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ScalarMulOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ScalarMulOp);
  ScalarMulOp() = default;
  ~ScalarMulOp() = default;

  void InitFromOpConf() override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().scalar_mul_conf(); }
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return true; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_SCALAR_MUL_OP_H_
