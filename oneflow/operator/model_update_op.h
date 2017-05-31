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
  const PbMessage& GetSpecialConf() const override;

 private:
  std::string ibn2lbn(const std::string& input_bn) const override {
    return kBaledBlobName;
  }
  std::string obn2lbn(const std::string& output_bn) const override {
    return kBaledBlobName;
  }

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_MODEL_UPDATE_OP_H_
