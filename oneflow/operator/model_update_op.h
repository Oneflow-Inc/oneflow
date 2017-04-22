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

  void Init(const OperatorConf& op_conf) override {
    mut_op_name() = op_conf.name();
    
    CHECK(op_conf.has_model_update_conf());
    auto cnf = new ModelUpdateOpConf(op_conf.model_update_conf());
    mut_pb_op_conf().reset(cnf);

    EnrollInputBn("model_diffs", false);
    EnrollInputBn("model_init", false);
    EnrollOutputBn("model", false);
  }

  std::string normal_ibn2lbn(const std::string& input_bn) const override {
    return RegstDesc::kAllLbn;
  }
  std::string obn2lbn(const std::string& output_bn) const override {
    return RegstDesc::kAllLbn;
  }

  void InferShape4ObAndDtbFromIb() const override { UNEXPECTED_RUN(); }

 private:
};

} // namespace oneflow

#endif // ONEFLOW_OPERATOR_MODEL_UPDATE_OP_H_
