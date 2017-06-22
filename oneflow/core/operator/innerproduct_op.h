#ifndef ONEFLOW_CORE_OPERATOR_INNERPRODUCT_OP_H_
#define ONEFLOW_CORE_OPERATOR_INNERPRODUCT_OP_H_

#include "oneflow/core/operator/operator_manager.h"

namespace oneflow {

class InnerProductOp final : public UserOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(InnerProductOp);
  InnerProductOp() = default;
  ~InnerProductOp() = default;

  void InitFromOpConf(const OperatorConf& op_conf) override;
  const PbMessage& GetSpecialConf() const override;
  void InferShape4FwBlobs(
      std::function<Shape*(const std::string&)> GetShapePtr4BnInOp,
      ParallelPolicy policy,
      uint64_t parallel_id,
      uint64_t parallel_num) const override;

 private:

};

} // namespace oneflow

#endif // ONEFLOW_CORE_OPERATOR_INNERPRODUCT_OP_H_
