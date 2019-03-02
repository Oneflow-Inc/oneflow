#ifndef ONEFLOW_CORE_OPERATOR_CALCULATE_SCALE_OP_H_
#define ONEFLOW_CORE_OPERATOR_CALCULATE_SCALE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class CalculateScaleOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CalculateScaleOp);
  CalculateScaleOp() = default;
  ~CalculateScaleOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override { return op_conf().calculate_scale_conf(); }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CALCULATE_SCALE_OP_H_
