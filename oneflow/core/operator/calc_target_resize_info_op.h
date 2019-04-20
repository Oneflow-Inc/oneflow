#ifndef ONEFLOW_CORE_OPERATOR_CALC_TARGET_RESIZE_INFO_OP_H_
#define ONEFLOW_CORE_OPERATOR_CALC_TARGET_RESIZE_INFO_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class CalcTargetResizeInfoOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CalcTargetResizeInfoOp);
  CalcTargetResizeInfoOp() = default;
  ~CalcTargetResizeInfoOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override {
    return op_conf().calc_target_resize_info_conf();
  }
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CALC_TARGET_RESIZE_INFO_OP_H_
