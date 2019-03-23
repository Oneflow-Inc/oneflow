#include "oneflow/core/optimizer/optimizer.h"

namespace oneflow {

namespace {

void GenerateOptimizerOpConf(
    const VariableOp& op, std::vector<OperatorConf>* op_confs,
    const std::function<const LogicalBlobId&(const std::string&)>& DiffLbi4BnInOp,
    const LogicalBlobId& total_loss_instance_num_lbi) {
  OperatorConf momentum_var(op.op_conf());
  momentum_var.set_name(op.op_name() + "-momentum");
  momentum_var.mutable_variable_conf()->set_out("out");
  op_confs->push_back(momentum_var);

  OperatorConf mdupdt_op;
  mdupdt_op.set_name(op.op_name() + "_optimizer");
  auto* mdupdt_op_conf = mdupdt_op.mutable_lars_model_update_conf();
  ConstructMdUpdtOpConf(op, DiffLbi4BnInOp, total_loss_instance_num_lbi, mdupdt_op_conf);
  mdupdt_op_conf->set_momentum(momentum_var.name() + "/out");
  op_confs->push_back(mdupdt_op);
}

}  // namespace

REGISTER_OPTIMIZER(NormalModelUpdateOpUserConf::kLarsConf, &GenerateOptimizerOpConf);

}  // namespace oneflow
