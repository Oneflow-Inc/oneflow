#include "oneflow/core/job_completer/op_graph_pass.h"
#include "oneflow/core/job/job.pb.h"

namespace oneflow {

namespace {

class AutoTrainStep final : public OpGraphPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AutoTrainStep);
  AutoTrainStep() = default;
  ~AutoTrainStep() override = default;

  bool IsEnabled() const override { return GlobalJobDesc().IsTrain(); }

  Maybe<void> Apply(const OpGraph& op_graph, Job* job) const override;
};

Maybe<void> AutoTrainStep::Apply(const OpGraph& op_graph, Job* job) const {
  if (job->job_conf().train_conf().has_train_step_lbn()) { return Maybe<void>::Ok(); }
  OperatorConf variable_op_conf{};
  const std::string train_step_name = "System-Train-TrainStep-" + job->job_conf().job_name();
  variable_op_conf.set_name(train_step_name);
  VariableOpConf* variable_conf = variable_op_conf.mutable_variable_conf();
  variable_conf->set_out("out");
  *variable_conf->mutable_shape()->mutable_dim()->Add() = 1;
  variable_conf->set_data_type(DataType::kInt64);
  variable_conf->mutable_split_axis()->clear_value();
  variable_conf->mutable_initializer()->mutable_constant_int_conf()->set_value(0);

  OperatorConf identity_op_conf{};
  identity_op_conf.set_name(train_step_name + "-Identity");
  IdentityOpConf* identity_conf = identity_op_conf.mutable_identity_conf();
  identity_conf->set_in(GenLogicalBlobName(variable_op_conf.name(), variable_conf->out()));
  identity_conf->set_out("out");
  const std::string& train_step_lbn =
      GenLogicalBlobName(identity_op_conf.name(), identity_conf->out());

  OperatorConf scalar_add_op_conf{};
  scalar_add_op_conf.set_name(train_step_name + "-ScalarAdd");
  ScalarAddOpConf* scalar_add_conf = scalar_add_op_conf.mutable_scalar_add_conf();
  scalar_add_conf->set_in(train_step_lbn);
  scalar_add_conf->set_out("out");
  scalar_add_conf->set_int_operand(1);

  OperatorConf assign_op_conf{};
  assign_op_conf.set_name(train_step_name + "-Assign");
  AssignOpConf* assign_conf = assign_op_conf.mutable_assign_conf();
  assign_conf->set_ref(GenLogicalBlobName(variable_op_conf.name(), variable_conf->out()));
  assign_conf->set_value(GenLogicalBlobName(scalar_add_op_conf.name(), scalar_add_conf->out()));

  JobBuilder job_builder(job);
  job_builder.AddOps(GenParallelConfOfCpuZeroOnMaster(),
                     {variable_op_conf, identity_op_conf, scalar_add_op_conf, assign_op_conf});
  job->mutable_job_conf()->mutable_train_conf()->set_train_step_lbn(train_step_lbn);
  return Maybe<void>::Ok();
}

REGISTER_FUNCTION_PASS("AutoTrainStep", AutoTrainStep);

}  // namespace

}  // namespace oneflow
