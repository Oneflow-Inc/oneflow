#ifndef ONEFLOW_OPERATOR_COPY_HD_OP_H_
#define ONEFLOW_OPERATOR_COPY_HD_OP_H_

#include "operator/operator.h"
#include "register/register_desc.h"

namespace oneflow {

class CopyHdOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyHdOp);
  CopyHdOp() = default;
  ~CopyHdOp() = default;

  void InitFromOpConf(const OperatorConf& op_conf) override;
  std::string GetValueFromPbOpConf(const std::string& k) const override;
  
 private:
  std::string ibn2lbn(const std::string& input_bn) const override {
    return RegstDesc::kAllLbn;
  }
  std::string obn2lbn(const std::string& output_bn) const override {
    return RegstDesc::kAllLbn;
  }


};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_COPY_HD_OP_H_
