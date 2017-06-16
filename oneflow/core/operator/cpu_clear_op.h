#ifndef ONEFLOW_CORE_OPERATOR_CPU_CLEAR_OP_H_
#define ONEFLOW_CORE_OPERATOR_CPU_CLEAR_OP_H_

#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/register_desc.h"

namespace oneflow {

class CpuClearOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuClearOp);
  CpuClearOp() = default;
  ~CpuClearOp() = default;

  void InitFromOpConf(const OperatorConf& op_conf) override;

  const PbMessage& GetSpecialConf() const override;
  
 private:
  std::string ibn2lbn(const std::string& input_bn) const override {
    return op_name();
  }
  std::string obn2lbn(const std::string& output_bn) const override {
    return op_name();
  }

};

} // namespace oneflow

#endif // ONEFLOW_CORE_OPERATOR_CPU_CLEAR_OP_H_
