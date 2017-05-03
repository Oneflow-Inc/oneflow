#ifndef ONEFLOW_OPERATOR_MODEL_LOAD_OP_H_
#define ONEFLOW_OPERATOR_MODEL_LOAD_OP_H_

#include "operator/operator.h"

namespace oneflow {

class ModelLoadOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelLoadOp);
  ModelLoadOp() = default;
  ~ModelLoadOp() = default;

  void InitFromOpConf(const OperatorConf& op_conf) override;
  std::string GetValueFromPbOpConf(const std::string& k) const override;

 private:
};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_MODEL_LOAD_OP_H_
