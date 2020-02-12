#include "oneflow/core/common/str_util.h"
#include "oneflow/core/job_completer/optimizer.h"

namespace oneflow {

namespace {

void GenerateOptimizerOpConf(const VariableOp& op, const ParallelConf& parallel_conf,
                             JobBuilder* job_builder, const LogicalBlobId& diff_lbi_of_var_out) {
  const std::string op_name = op.op_name() + "-momentum";
  OperatorConf momentum_var(op.op_conf());
  const bool has_snapshot_path =
      job_builder->job().job_conf().has_default_initialize_with_snapshot_path();
  std::string file_path = "";
  if (has_snapshot_path) {
    file_path = JoinPath(job_builder->job().job_conf().default_initialize_with_snapshot_path(),
                         op_name, "out");
  }
  if (has_snapshot_path && SnapshotFS()->FileExists(file_path)) {
    LOG(INFO) << "file_path: " << file_path;
    momentum_var.mutable_variable_conf()->mutable_initialize_with_snapshot()->set_path(
        JoinPath(job_builder->job().job_conf().default_initialize_with_snapshot_path(), op_name));
    momentum_var.mutable_variable_conf()->mutable_initialize_with_snapshot()->set_key("out");
  } else {
    if (has_snapshot_path) { LOG(INFO) << file_path << " not found, will be initialized"; }
    InitializerConf constant_initializer;
    constant_initializer.mutable_constant_conf()->set_value(0.f);
    *(momentum_var.mutable_variable_conf()->mutable_initializer()) = constant_initializer;
  }
  momentum_var.set_name(op_name);
  momentum_var.mutable_variable_conf()->set_out("out");
  job_builder->AddOps(parallel_conf, {momentum_var});

  OperatorConf mdupdt_op;
  mdupdt_op.set_name(op.op_name() + "_optimizer");
  auto* mdupdt_op_conf = mdupdt_op.mutable_momentum_model_update_conf();
  ConstructMdUpdtOpConf(op, diff_lbi_of_var_out, job_builder, mdupdt_op_conf);
  mdupdt_op_conf->set_momentum(momentum_var.name() + "/out");
  job_builder->AddOps(parallel_conf, {mdupdt_op});
}

}  // namespace

REGISTER_OPTIMIZER(NormalModelUpdateOpUserConf::kMomentumConf, &GenerateOptimizerOpConf);

}  // namespace oneflow
