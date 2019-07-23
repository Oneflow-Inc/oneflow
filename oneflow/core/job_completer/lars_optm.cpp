#include "oneflow/core/job_completer/optimizer.h"

namespace oneflow {

namespace {

void GenerateOptimizerOpConf(const VariableOp& op, const ParallelConf& parallel_conf,
                             JobBuilder* job_builder, const LogicalBlobId& diff_lbi_of_var_out,
                             const LogicalBlobId& total_loss_instance_num_lbi) {
  OperatorConf momentum_var(op.op_conf());
  momentum_var.set_name(op.op_name() + "-momentum");
  momentum_var.mutable_variable_conf()->set_out("out");
  job_builder->AddOps(parallel_conf, {momentum_var});
  BindTwoVariableOpObnSbpConf(momentum_var.name(), op.op_name(), job_builder);

  OperatorConf mdupdt_op;
  mdupdt_op.set_name(op.op_name() + "_optimizer");
  auto* mdupdt_op_conf = mdupdt_op.mutable_lars_model_update_conf();
  ConstructMdUpdtOpConf(op, diff_lbi_of_var_out, total_loss_instance_num_lbi, mdupdt_op_conf);
  mdupdt_op_conf->set_momentum(momentum_var.name() + "/out");
  job_builder->AddOps(parallel_conf, {mdupdt_op});
}

}  // namespace

REGISTER_OPTIMIZER(NormalModelUpdateOpUserConf::kLarsConf, &GenerateOptimizerOpConf);

}  // namespace oneflow
