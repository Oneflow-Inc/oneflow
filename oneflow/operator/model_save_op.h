#ifndef ONEFLOW_OPERATOR_MODEL_SAVE_OP_H_
#define ONEFLOW_OPERATOR_MODEL_SAVE_OP_H_

#include "operator/operator.h"
#include "register/register_desc.h"

namespace oneflow {

class ModelSaveOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSaveOp);
  ModelSaveOp() = default;
  ~ModelSaveOp() = default;

  void InitFromOpConf(const OperatorConf& op_conf) override;
  const PbMessage& GetSpecialConf() const override;

 private:
  std::string ibn2lbn(const std::string& ibn) const override {
    return RegstDesc::kAllLbn;
  }

};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_MODEL_SAVE_OP_H_
