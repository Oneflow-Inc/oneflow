#ifndef ONEFLOW_OPERATOR_SPLIT_OP_H_
#define ONEFLOW_OPERATOR_SPLIT_OP_H_

#include "operator/operator.h"

namespace oneflow {

class SplitOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SplitOp);
  SplitOp() = default;
  ~SplitOp() = default;

  void Init(const OperatorConf& op_conf) override;
  void InferShape4ObAndDtbFromIb() const override { TODO(); }
  void InferShape4IbAndDtbFromOb() const override { TODO(); }
  
  std::string ibn2lbn(const std::string& input_bn) const override {
    return GetValueFromPbOpConf("lbn");
  }
  std::string obn2lbn(const std::string& output_bn) const override {
    return GetValueFromPbOpConf("lbn");
  }

 private:

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_SPLIT_OP_H_
