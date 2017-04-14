#ifndef ONEFLOW_OPERATOR_CONCAT_OP_H_
#define ONEFLOW_OPERATOR_CONCAT_OP_H_

#include "operator/operator.h"

namespace oneflow {

class ConcatOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ConcatOp);
  ConcatOp() = default;
  ~ConcatOp() = default;

  void Init(const OperatorConf& op_conf) override;
  void InferShape4ObAndDtbFromIb() const override { TODO(); }
  
  std::string normal_ibn2lbn(const std::string& input_bn) const override {
    return GetValueFromPbOpConf("lbn");
  }
  std::string obn2lbn(const std::string& output_bn) const override {
    return GetValueFromPbOpConf("lbn");
  }

 private:

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_CONCAT_OP_H_
