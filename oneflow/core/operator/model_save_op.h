#ifndef ONEFLOW_CORE_OPERATOR_MODEL_SAVE_OP_H_
#define ONEFLOW_CORE_OPERATOR_MODEL_SAVE_OP_H_

#include "oneflow/core/operator/operator.h"

namespace oneflow {

class ModelSaveOp final : public Operator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSaveOp);
  ModelSaveOp() = default;
  ~ModelSaveOp() = default;

  void InitFromOpConf() override;
  const PbMessage& GetCustomizedConf() const override;

 private:
  void InferOutputBlobModelSplitAxis(
      std::function<int32_t*(const std::string&)> ModelSplitAxis4BnInOp,
      std::function<int32_t(const std::string&)> ShapeNumAxes4BnInOp,
      const ParallelContext* parallel_context) const override {
    UNIMPLEMENTED();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OPERATOR_MODEL_SAVE_OP_H_
