#ifndef ONEFLOW_OPERATOR_BOXING_OP_H_
#define ONEFLOW_OPERATOR_BOXING_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class BoxingOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingOp);
  BoxingOp() = default;
  ~BoxingOp() = default;

  void InitFromOpConf(const OperatorConf& op_conf) override;
  const PbMessage& GetSpecialConf() const override;

  void InferShape4FwBlobs(
      std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
      ParallelPolicy policy,
      uint64_t parallel_id,
      uint64_t parallel_num) const override;
  
 private:
  std::string ibn2lbn(const std::string& input_bn) const override;
  std::string obn2lbn(const std::string& output_bn) const override;

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_BOXING_OP_H_
