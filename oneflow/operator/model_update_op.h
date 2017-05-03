#ifndef ONEFLOW_OPERATOR_MODEL_UPDATE_OP_H_
#define ONEFLOW_OPERATOR_MODEL_UPDATE_OP_H_

#include "operator/operator.h"
#include "register/register_desc.h"

namespace oneflow {

class ModelUpdateOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelUpdateOp);
  ModelUpdateOp() = default;
  ~ModelUpdateOp() = default;

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

#endif // ONEFLOW_OPERATOR_MODEL_UPDATE_OP_H_
