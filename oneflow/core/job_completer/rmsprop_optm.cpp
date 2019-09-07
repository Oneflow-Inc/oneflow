#include "oneflow/core/job_completer/optimizer.h"

namespace oneflow {

namespace {

void GenerateOptimizerOpConf(const VariableOp& op, const ParallelConf& parallel_conf,
                             JobBuilder* job_builder, const LogicalBlobId& diff_lbi_of_var_out,
                             const LogicalBlobId& total_loss_instance_num_lbi) {
  OperatorConf mdupdt_op;
  mdupdt_op.set_name(op.op_name() + "_optimizer");
  ConstructMdUpdtOpConf(op, diff_lbi_of_var_out, total_loss_instance_num_lbi, job_builder,
                        mdupdt_op.mutable_rmsprop_model_update_conf());
  job_builder->AddOps(parallel_conf, {mdupdt_op});
}

}  // namespace

REGISTER_OPTIMIZER(NormalModelUpdateOpUserConf::kRmspropConf, &GenerateOptimizerOpConf);

}  // namespace oneflow
