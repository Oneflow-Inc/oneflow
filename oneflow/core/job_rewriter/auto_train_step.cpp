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

class AutoTrainStep final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AutoTrainStep);
  AutoTrainStep() = default;
  ~AutoTrainStep() override = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override;
};

Maybe<void> AutoTrainStep::Apply(Job* job, JobPassCtx* ctx) const {
  if (!ctx->job_desc().IsTrain()) { return Maybe<void>::Ok(); }
  const OpGraph op_graph(*job);
  const TrainConf& train_conf = job->job_conf().train_conf();
  if (train_conf.has_train_step_lbn()) {
    CHECK_OR_RETURN(!train_conf.has_dynamic_loss_scale_policy());
    return Maybe<void>::Ok();
  }
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

  JobBuilder job_builder(job);
  const ParallelConf& parallel_conf = GenParallelConfOfCpuZeroOnMaster();
  int64_t scope_symbol_id = 0;
  {
    const std::shared_ptr<cfg::JobConfigProto>& cfg_job_conf =
        std::make_shared<cfg::JobConfigProto>(job->job_conf());
    const std::shared_ptr<cfg::ParallelConf>& cfg_parallel_conf =
        std::make_shared<cfg::ParallelConf>(parallel_conf);
    scope_symbol_id =
        Global<ForeignCallback>::Get()->MakeScopeSymbol(cfg_job_conf, cfg_parallel_conf, false);
  }

  auto scalar_add_op = user_op::UserOpConfWrapperBuilder(train_step_name + "-ScalarAdd")
                           .Op("scalar_add")
                           .Input("in", train_step_lbn)
                           .Output("out")
                           .Attr<bool>("has_float_operand", false)
                           .Attr<double>("float_operand", 0)
                           .Attr<bool>("has_int_operand", true)
                           .Attr<int64_t>("int_operand", 1)
                           .ScopeSymbolId(scope_symbol_id)
                           .Build();

  variable_op_conf.set_scope_symbol_id(scope_symbol_id);
  identity_op_conf.set_scope_symbol_id(scope_symbol_id);
  job_builder.AddOps(parallel_conf, {variable_op_conf, identity_op_conf, scalar_add_op.op_conf()});
  if (train_conf.has_dynamic_loss_scale_policy()) {
    const auto& dynamic_loss_scale_state =
        JUST(ctx->GetState<DynamicLossScaleJobPassState>("dynamic_loss_scale_state"));
    auto assign_op =
        user_op::UserOpConfWrapperBuilder(train_step_name + "-AssignIfNot")
            .Op("assign_if_not")
            .Input("ref", GenLogicalBlobName(variable_op_conf.name(), variable_conf->out()))
            .Input("value", scalar_add_op.output("out", 0))
            .Input("condition", dynamic_loss_scale_state.count_not_finite_lbn())
            .ScopeSymbolId(scope_symbol_id)
            .Build();
    job_builder.AddOps(parallel_conf, {assign_op.op_conf()});
  } else {
    auto assign_op =
        user_op::UserOpConfWrapperBuilder(train_step_name + "-Assign")
            .Op("assign")
            .Input("ref", GenLogicalBlobName(variable_op_conf.name(), variable_conf->out()))
            .Input("value", scalar_add_op.output("out", 0))
            .ScopeSymbolId(scope_symbol_id)
            .Build();
    job_builder.AddOps(parallel_conf, {assign_op.op_conf()});
  }

  job->mutable_job_conf()->mutable_train_conf()->set_train_step_lbn(train_step_lbn);
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("AutoTrainStep", AutoTrainStep);

}  // namespace

}  // namespace oneflow
