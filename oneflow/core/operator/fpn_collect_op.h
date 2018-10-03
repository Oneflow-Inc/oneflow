#ifndef ONEFLOW_CORE_OPERATOR_FPN_COLLECT_OP_H
#define ONEFLOW_CORE_OPERATOR_FPN_COLLECT_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class FpnCollectOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(FpnCollectOp);
  FpnCollectOp() = default;
  ~FpnCollectOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedInBlobWhenBackward() const override { return false; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_FPN_COLLECT_OP_H_
