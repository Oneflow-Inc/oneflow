#ifndef ONEFLOW_CORE_OPERATOR_ONE_HOT_H_
#define ONEFLOW_CORE_OPERATOR_ONE_HOT_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class OneHot final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OneHot);
  OneHot() = default;
  ~OneHot() override = default;
  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  bool NeedOutBlobWhenBackward() const override { return false; }
  bool NeedInBlobWhenBackward() const override { return false; }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ONE_HOT_H_
