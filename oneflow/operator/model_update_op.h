#ifndef ONEFLOW_OPERATOR_MODEL_UPDATE_OP_H_
#define ONEFLOW_OPERATOR_MODEL_UPDATE_OP_H_

#include "operator/operator.h"

namespace oneflow {

class ModelUpdateOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelUpdateOp);
  ModelUpdateOp() default;
  ~ModelUpdateOp() = default;

  void Init(const OperatorConf& op_conf) override {
    mut_op_name() = op_conf.name();
    
    CHECK(op_conf.has_model_update_op_conf());
    auto cnf = new ModelUpdateOp(op_conf.model_update_op_conf());
    mut_pb_op_conf().reset(cnf);

    EnrollInputBn("model_diffs", false);
    EnrollOutputBn("model", false);
  }
  void InferShape4ObAndDtbFromIb() const override { TODO(); }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_MODEL_UPDATE_OP_H_
