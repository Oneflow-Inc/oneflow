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
#include "oneflow/core/common/util.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

class AutoLearningRate final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AutoLearningRate);
  AutoLearningRate() = default;
  ~AutoLearningRate() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const { return ctx.job_desc().IsTrain(); }

  Maybe<void> Apply(const OpGraph& op_graph, Job* job) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    return Apply(op_graph, job);
  }
};

Maybe<void> AutoLearningRate::Apply(const OpGraph& op_graph, Job* job) const {
  JobBuilder job_builder(job);
  const TrainConf& train_conf = job->job_conf().train_conf();
  auto AddScheduleOp = [&](const OptimizerConf& optimizer_conf,
                           const std::string& op_name) -> std::string {
    const class oneflow::OpNode* op_node =
        op_graph.OpNode4OpName(GenLogicalBlobId(train_conf.train_step_lbn()).op_name());
    CHECK_OR_RETURN(op_node != nullptr) << "op node not found in op graph, op name: " << op_name;
    const ParallelConf& parallel_conf = op_node->parallel_desc().parallel_conf();
    if (optimizer_conf.has_warmup_conf() || optimizer_conf.has_learning_rate_decay()) {
      OperatorConf schedule_op_conf{};
      schedule_op_conf.set_name(op_name);
      LearningRateScheduleOpConf* schedule_conf =
          schedule_op_conf.mutable_learning_rate_schedule_conf();
      schedule_conf->set_train_step(train_conf.train_step_lbn());
      schedule_conf->set_learning_rate(optimizer_conf.base_learning_rate());
      schedule_conf->set_out("out");
      if (optimizer_conf.has_warmup_conf()) {
        *schedule_conf->mutable_warmup_conf() = optimizer_conf.warmup_conf();
      }
      if (optimizer_conf.has_learning_rate_decay()) {
        *schedule_conf->mutable_learning_rate_decay() = optimizer_conf.learning_rate_decay();
      }
      schedule_op_conf.set_scope_symbol_id(op_node->op().op_conf().scope_symbol_id());
      job_builder.AddOps(parallel_conf, {schedule_op_conf});
      return GenLogicalBlobName(op_name, schedule_conf->out());
    } else {
      const auto constant_op =
          user_op::UserOpConfWrapperBuilder(op_name)
              .Op("constant")
              .Attr<double>("floating_value", optimizer_conf.base_learning_rate())
              .Attr<int64_t>("integer_value", 0)
              .Attr<bool>("is_floating_value", true)
              .Attr<DataType>("dtype", DataType::kFloat)
              .Attr<Shape>("shape", Shape({1}))
              .Output("out")
              .ScopeSymbolId(op_node->op().op_conf().scope_symbol_id())
              .Build();
      job_builder.AddOps(parallel_conf, {constant_op.op_conf()});
      return constant_op.output("out", 0);
    }
  };
  FOR_RANGE(int64_t, i, 0, train_conf.optimizer_conf_size()) {
    const auto& optimizer_conf = train_conf.optimizer_conf(i);
    const std::string& lbn =
        AddScheduleOp(optimizer_conf, "System-Train-LearningRate-Scheduler_" + NewUniqueId());
    job->mutable_job_conf()->mutable_train_conf()->mutable_optimizer_conf(i)->set_learning_rate_lbn(
        lbn);
  }
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("AutoLearningRate", AutoLearningRate);

}  // namespace

}  // namespace oneflow
