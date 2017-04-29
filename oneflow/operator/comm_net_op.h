#ifndef ONEFLOW_OPERATOR_COMM_NET_OP_H_
#define ONEFLOW_OPERATOR_COMM_NET_OP_H_

#include "operator/operator.h"
#include "register/register_desc.h"

namespace oneflow {

class CommNetOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CommNetOp);
  CommNetOp() = default;
  ~CommNetOp() = default;

  void InitFromOpConf(const OperatorConf& op_conf) override;
  void InferShape4ObAndDtbFromIb() const override { UNEXPECTED_RUN(); }
  std::string GetValueFromPbOpConf(const std::string& k) const override;
  
  std::string normal_ibn2lbn(const std::string& input_bn) const override;
  std::string obn2lbn(const std::string& output_bn) const override;

 private:
};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_Comm_Net_OP_H_
