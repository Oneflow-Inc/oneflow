#ifndef ONEFLOW_CORE_OPERATOR_IDENTITY_OP_H_
#define ONEFLOW_CORE_OPERATOR_IDENTITY_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class IdentityOp : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(IdentityOp);
  IdentityOp() = default;
  virtual ~IdentityOp() = default;

  void InitFromOpConf() override;
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_IDENTITY_OP_H_
