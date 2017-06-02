#ifndef ONEFLOW_OPERATOR_COPY_COMM_NET_OP_H_
#define ONEFLOW_OPERATOR_COPY_COMM_NET_OP_H_

#include "oneflow/operator/operator.h"
#include "oneflow/register/register_desc.h"

namespace oneflow {

class CopyCommNetOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyCommNetOp);
  CopyCommNetOp() = default;
  ~CopyCommNetOp() = default;

  void InitFromOpConf(const OperatorConf& op_conf) override;
  const PbMessage& GetSpecialConf() const override;
  
 private:
  std::string ibn2lbn(const std::string& input_bn) const override;
  std::string obn2lbn(const std::string& output_bn) const override;

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_COPY_COMM_NET_OP_H_
