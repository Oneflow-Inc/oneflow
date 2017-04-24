#ifndef ONEFLOW_OPERATOR_MODEL_LOAD_OP_H_
#define ONEFLOW_OPERATOR_MODEL_LOAD_OP_H_

#include "operator/operator.h"

namespace oneflow {

class ModelLoadOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelLoadOp);
  ModelLoadOp() = default;
  ~ModelLoadOp() = default;

  std::string GetValueFromPbOpConf(const std::string& k) const override;
  void InitFromOpConf(const OperatorConf& op_conf) override;
  void InferShape4ObAndDtbFromIb() const override { TODO(); }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_MODEL_LOAD_OP_H_
