#ifndef ONEFLOW_OPERATOR_MODEL_LOAD_OP_H_
#define ONEFLOW_OPERATOR_MODEL_LOAD_OP_H_

#include "operator/operator.h"

namespace oneflow {

class ModelLoadOp final : public SysOperator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelLoadOp);
  ModelLoadOp() default;
  ~ModelLoadOp() = default;

  void Init(const OperatorConf& op_conf) override {
    mut_op_name() = op_conf.name();
    
    CHECK(op_conf.has_model_load_op_conf());
    auto cnf = new ModelLoadOp(op_conf.model_load_op_conf());
    mut_pb_op_conf().reset(cnf);

    EnrollOutputBn("model");
  }
  void InferShape4ObAndDtbFromIb() const override { TODO(); }
  void InferShape4IbAndDtbFromOb() const override { TODO(); }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_MODEL_LOAD_OP_H_
