#ifndef ONEFLOW_OPERATOR_COPY_OP_H_
#define ONEFLOW_OPERATOR_COPY_OP_H_

#include "operator/operator.h"

namespace oneflow {

class CopyOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyOp);
  CopyOp() = default;
  ~CopyOp() = default;

  void Init(const OperatorConf& op_conf) override;
  void InferShape4ObAndDtbFromIb() const override { TODO(); }
  
  std::string ibn2lbn(const std::string& input_bn) const override {
    return ibn2lbn_.at(input_bn);
  }
  std::string obn2lbn(const std::string& output_bn) const override {
    return obn2lbn_.at(output_bn);
  }

 private:
  HashMap<std::string, std::string> ibn2lbn_;
  HashMap<std::string, std::string> obn2lbn_;

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_COPY_OP_H_
