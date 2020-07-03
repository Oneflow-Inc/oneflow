#include "oneflow/core/common/util.h"
#include "oneflow/core/job_completer/op_graph_pass.h"
#include "oneflow/core/job/job.pb.h"

namespace oneflow {

namespace {

class AutoLearningRate final : public OpGraphPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AutoLearningRate);
  AutoLearningRate() = default;
  ~AutoLearningRate() override = default;

  bool IsEnabled() const override { return GlobalJobDesc().IsTrain(); }

  Maybe<void> Apply(const OpGraph& op_graph, Job* job) const override;
};

Maybe<void> AutoLearningRate::Apply(const OpGraph& op_graph, Job* job) const {
  JobBuilder job_builder(job);
  const TrainConf& train_conf = job->job_conf().train_conf();
  auto AddScheduleOp = [&](const std::string& op_name, const float learning_rate) -> std::string {
    const class oneflow::OpNode* op_node =
        op_graph.OpNode4OpName(GenLogicalBlobId(train_conf.train_step_lbn()).op_name());
    CHECK_OR_RETURN(op_node != nullptr) << "op node not fould in op graph, op name: " << op_name;
    const ParallelConf& parallel_conf = op_node->parallel_desc().parallel_conf();
    const NormalModelUpdateOpUserConf& model_update_conf = train_conf.model_update_conf();
    if (model_update_conf.has_warmup_conf() || model_update_conf.has_learning_rate_decay()) {
      OperatorConf schedule_op_conf{};
      schedule_op_conf.set_name(op_name);
      LearningRateScheduleOpConf* schedule_conf =
          schedule_op_conf.mutable_learning_rate_schedule_conf();
      schedule_conf->set_train_step(train_conf.train_step_lbn());
      schedule_conf->set_learning_rate(learning_rate);
      schedule_conf->set_out("out");
      if (model_update_conf.has_warmup_conf()) {
        *schedule_conf->mutable_warmup_conf() = model_update_conf.warmup_conf();
      }
      if (model_update_conf.has_learning_rate_decay()) {
        *schedule_conf->mutable_learning_rate_decay() = model_update_conf.learning_rate_decay();
      }
      job_builder.AddOps(parallel_conf, {schedule_op_conf});
      return GenLogicalBlobName(op_name, schedule_conf->out());
    } else {
      OperatorConf constant_op_conf{};
      constant_op_conf.set_name(op_name);
      ConstantOpConf* constant_conf = constant_op_conf.mutable_constant_conf();
      constant_conf->set_out("out");
      *constant_conf->mutable_shape()->mutable_dim()->Add() = 1;
      constant_conf->set_data_type(DataType::kFloat);
      constant_conf->mutable_initializer()->mutable_constant_conf()->set_value(learning_rate);
      job_builder.AddOps(parallel_conf, {constant_op_conf});
      return GenLogicalBlobName(op_name, constant_conf->out());
    }
  };
  if (!train_conf.has_primary_lr_lbn()) {
    CHECK(train_conf.has_primary_lr());
    const std::string lbn =
        AddScheduleOp("System-Train-PrimaryLearningRate-Scheduler", train_conf.primary_lr());
    job->mutable_job_conf()->mutable_train_conf()->set_primary_lr_lbn(lbn);
  }
  if (!train_conf.has_secondary_lr_lbn()) {
    if (train_conf.has_secondary_lr()) {
      const std::string lbn =
          AddScheduleOp("System-Train-SecondaryLearningRate-Scheduler", train_conf.secondary_lr());
      job->mutable_job_conf()->mutable_train_conf()->set_secondary_lr_lbn(lbn);
    } else {
      job->mutable_job_conf()->mutable_train_conf()->set_secondary_lr_lbn(
          train_conf.primary_lr_lbn());
    }
  }
  return Maybe<void>::Ok();
}

REGISTER_FUNCTION_PASS("AutoLearningRate", AutoLearningRate);

}  // namespace

}  // namespace oneflow
