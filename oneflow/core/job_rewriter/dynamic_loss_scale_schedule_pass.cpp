/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/foreign_callback.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/job_rewriter/dynamic_loss_scale_job_pass_state.h"

namespace oneflow {

namespace {

class DynamicLossScaleSchedulePass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DynamicLossScaleSchedulePass);
  DynamicLossScaleSchedulePass() = default;
  ~DynamicLossScaleSchedulePass() override = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

Maybe<void> DynamicLossScaleSchedulePass::Apply(Job* job, JobPassCtx* ctx) const {
  if (!ctx->job_desc().IsTrain()) { return Maybe<void>::Ok(); }
  const TrainConf& train_conf = job->job_conf().train_conf();
  if (!train_conf.has_dynamic_loss_scale_policy()) { return Maybe<void>::Ok(); }
  const auto& policy = train_conf.dynamic_loss_scale_policy();
  const OpGraph op_graph(*job);
  JobBuilder job_builder(job);
  const ParallelConf& parallel_conf = GenParallelConfOfCpuZeroOnMaster();
  int64_t scope_symbol_id;
  {
    const std::shared_ptr<cfg::JobConfigProto>& cfg_job_conf =
        std::make_shared<cfg::JobConfigProto>(job->job_conf());
    const std::shared_ptr<cfg::ParallelConf>& cfg_parallel_conf =
        std::make_shared<cfg::ParallelConf>(parallel_conf);
    scope_symbol_id = (*Global<std::shared_ptr<ForeignCallback>>::Get())
                          ->MakeScopeSymbol(cfg_job_conf, cfg_parallel_conf, false);
  }
  OperatorConf loss_scale_var_op_conf{};
  const std::string op_name_prefix = "System-Train-DynamicLossScale-";
  {
    loss_scale_var_op_conf.set_name(op_name_prefix + job->job_conf().job_name() + "-LossScale");
    VariableOpConf* variable_conf = loss_scale_var_op_conf.mutable_variable_conf();
    variable_conf->set_out("out");
    *variable_conf->mutable_shape()->mutable_dim()->Add() = 1;
    variable_conf->set_data_type(DataType::kFloat);
    *variable_conf->add_parallel_distribution() = "B";
    variable_conf->mutable_initializer()->mutable_constant_conf()->set_value(
        policy.initial_loss_scale());
    loss_scale_var_op_conf.set_scope_symbol_id(scope_symbol_id);
  }
  OperatorConf good_step_counter_var_conf{};
  {
    good_step_counter_var_conf.set_name(op_name_prefix + job->job_conf().job_name()
                                        + "-GoodStepCounter");
    VariableOpConf* variable_conf = good_step_counter_var_conf.mutable_variable_conf();
    variable_conf->set_out("out");
    *variable_conf->mutable_shape()->mutable_dim()->Add() = 1;
    variable_conf->set_data_type(DataType::kInt64);
    *variable_conf->add_parallel_distribution() = "B";
    variable_conf->mutable_initializer()->mutable_constant_int_conf()->set_value(0);
    good_step_counter_var_conf.set_scope_symbol_id(scope_symbol_id);
  }
  OperatorConf loss_scale_val_op_conf{};
  const std::string loss_scale_var_lbn = GenLogicalBlobName(
      loss_scale_var_op_conf.name(), loss_scale_var_op_conf.variable_conf().out());
  {
    loss_scale_val_op_conf.set_name(loss_scale_var_op_conf.name() + "-Identity");
    loss_scale_val_op_conf.set_scope_symbol_id(scope_symbol_id);
    IdentityOpConf* identity_conf = loss_scale_val_op_conf.mutable_identity_conf();
    identity_conf->set_in(loss_scale_var_lbn);
    identity_conf->set_out("out");
  }
  // will be replaced by real count of not finite
  auto count_not_finite_stub_op =
      user_op::UserOpConfWrapperBuilder(op_name_prefix + job->job_conf().job_name()
                                        + "-CountNotFinite")
          .Op("constant")
          .Output("out")
          .Attr<double>("floating_value", 0.0)
          .Attr<int64_t>("integer_value", 0)
          .Attr<bool>("is_floating_value", false)
          .Attr<DataType>("dtype", DataType::kInt64)
          .Attr<Shape>("shape", Shape({1}))
          .ScopeSymbolId(scope_symbol_id)
          .Build();
  const std::string loss_scale_val_lbn = GenLogicalBlobName(
      loss_scale_val_op_conf.name(), loss_scale_val_op_conf.identity_conf().out());
  const std::string good_step_counter_var_lbn = GenLogicalBlobName(
      good_step_counter_var_conf.name(), good_step_counter_var_conf.variable_conf().out());
  auto schedule =
      user_op::UserOpConfWrapperBuilder(op_name_prefix + job->job_conf().job_name() + "-Schedule")
          .Op("dynamic_loss_scale_schedule")
          .Input("count_not_finite", count_not_finite_stub_op.output("out", 0))
          .Input("loss_scale", loss_scale_var_lbn)
          .Input("good_step_counter", good_step_counter_var_lbn)
          .Attr<int64_t>("increment_period", policy.increment_period())
          .Attr<float>("multiplier", policy.multiplier())
          .ScopeSymbolId(scope_symbol_id)
          .Build();
  job_builder.AddOps(parallel_conf,
                     {loss_scale_var_op_conf, loss_scale_val_op_conf, good_step_counter_var_conf,
                      count_not_finite_stub_op.op_conf(), schedule.op_conf()});
  if (!JUST(ctx->HasState<DynamicLossScaleJobPassState>("dynamic_loss_scale_state"))) {
    ctx->ResetState("dynamic_loss_scale_state", std::make_unique<DynamicLossScaleJobPassState>());
  }
  auto state = JUST(ctx->MutableState<DynamicLossScaleJobPassState>("dynamic_loss_scale_state"));
  state->set_loss_scale_val_lbn(loss_scale_val_lbn);
  state->set_count_not_finite_lbn(count_not_finite_stub_op.output("out", 0));
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("DynamicLossScaleSchedulePass", DynamicLossScaleSchedulePass);

}  // namespace

}  // namespace oneflow
