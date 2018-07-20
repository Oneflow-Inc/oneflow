#ifndef ONEFLOW_CORE_OPERATOR_ROI_POOLING_OP_H_
#define ONEFLOW_CORE_OPERATOR_ROI_POOLING_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class RoIPoolingOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RoIPoolingOp);
  RoIPoolingOp() = default;
  virtual ~RoIPoolingOp() = default;

  const PbMessage& GetCustomizedConf() const override;

  void InitFromOpConf() override;

  bool NeedOutWhenBackward() const override { return false; }

  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_ROI_POOLING_OP_H_
