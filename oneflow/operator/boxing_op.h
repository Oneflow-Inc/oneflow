#ifndef ONEFLOW_OPERATOR_BOXING_OP_H_
#define ONEFLOW_OPERATOR_BOXING_OP_H_

#include "operator/operator.h"

namespace oneflow {

const int32_t gDataConcatInitAxis = -1;
const int32_t gModelConcatInitAxis = -2;

class BoxingOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingOp);
  BoxingOp() = default;
  ~BoxingOp() = default;

  void InitFromOpConf(const OperatorConf& op_conf) override;
  const PbMessage& GetSpecialConf() const override;
  
 private:
  std::string ibn2lbn(const std::string& input_bn) const override;
  std::string obn2lbn(const std::string& output_bn) const override;

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_BOXING_OP_H_
