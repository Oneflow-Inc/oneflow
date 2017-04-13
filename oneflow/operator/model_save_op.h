#ifndef ONEFLOW_OPERATOR_MODEL_SAVE_OP_H_
#define ONEFLOW_OPERATOR_MODEL_SAVE_OP_H_

#include "operator/operator.h"

namespace oneflow {

class ModelSaveOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelSaveOp);
  ModelSaveOp() default;
  ~ModelSaveOp() = default;

  void Init(const OperatorConf& op_conf) override {
    mut_op_name() = op_conf.name();
    
    CHECK(op_conf.has_model_save_op_conf());
    auto cnf = new ModelSaveOp(op_conf.model_save_op_conf());
    mut_pb_op_conf().reset(cnf);

    EnrollInputBn("model");
  }
  void InferBlobDesc4ObAndDtbFromIb() const override { TODO(); }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_MODEL_SAVE_OP_H_
