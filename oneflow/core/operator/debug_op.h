#ifndef ONEFLOW_CORE_OPERATOR_DEBUG_OP_H_
#define ONEFLOW_CORE_OPERATOR_DEBUG_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class DebugOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DebugOp);
  DebugOp() = default;
  ~DebugOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().debug_conf(); }
  bool IsElemWiseOp() const override { return true; }
  bool NeedInBlobWhenBackward() const override { return false; }
  bool NeedOutBlobWhenBackward() const override { return false; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_DEBUG_OP_H_
