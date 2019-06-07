#ifndef ONEFLOW_CORE_OPERATOR_CALC_IOU_MATRIX_OP_H_
#define ONEFLOW_CORE_OPERATOR_CALC_IOU_MATRIX_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class CalcIoUMatrixOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CalcIoUMatrixOp);
  CalcIoUMatrixOp() = default;
  ~CalcIoUMatrixOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;
  void InferBlobDescs(std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
                      const ParallelContext* parallel_ctx) const override;

 private:
  bool IsInputBlobAllowedModelSplit(const std::string& ibn) const override { return false; }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_CALC_IOU_MATRIX_OP_H_
