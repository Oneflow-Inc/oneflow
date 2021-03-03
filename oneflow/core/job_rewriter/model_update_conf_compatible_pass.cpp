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

class ModelUpdateConfCompatiblePass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ModelUpdateConfCompatiblePass);
  ModelUpdateConfCompatiblePass() = default;
  ~ModelUpdateConfCompatiblePass() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const { return ctx.job_desc().IsTrain(); }

  Maybe<void> Apply(const OpGraph& op_graph, Job* job) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    return Apply(op_graph, job);
  }
};

Maybe<void> ModelUpdateConfCompatiblePass::Apply(const OpGraph& op_graph, Job* job) const {
  JobBuilder job_builder(job);
  const TrainConf& train_conf = job->job_conf().train_conf();
  if (!train_conf.has_model_update_conf()) { return Maybe<void>::Ok(); }
  CHECK_OR_RETURN(train_conf.optimizer_conf_size() == 0);
  const NormalModelUpdateOpUserConf& model_update_conf = train_conf.model_update_conf();
  OptimizerConf* optimizer_conf =
      job->mutable_job_conf()->mutable_train_conf()->add_optimizer_conf();
  op_graph.ForEachNode([&](OpNode* op_node) {
    if (op_node->op().op_conf().has_variable_conf()) {
      optimizer_conf->add_variable_op_names(op_node->op().op_name());
    }
  });
  if (model_update_conf.has_learning_rate_decay()) {
    optimizer_conf->mutable_learning_rate_decay()->CopyFrom(
        model_update_conf.learning_rate_decay());
  }
  if (model_update_conf.has_warmup_conf()) {
    optimizer_conf->mutable_warmup_conf()->CopyFrom(model_update_conf.warmup_conf());
  }
  if (model_update_conf.has_clip_conf()) {
    optimizer_conf->mutable_clip_conf()->CopyFrom(model_update_conf.clip_conf());
  }
  if (model_update_conf.has_weight_decay_conf()) {
    optimizer_conf->mutable_weight_decay_conf()->CopyFrom(model_update_conf.weight_decay_conf());
  }
  if (train_conf.has_primary_lr()) {
    optimizer_conf->set_base_learning_rate(train_conf.primary_lr());
  }
  if (train_conf.has_primary_lr_lbn()) {
    optimizer_conf->set_learning_rate_lbn(train_conf.primary_lr_lbn());
  }
  if (model_update_conf.has_naive_conf()) {
    optimizer_conf->mutable_naive_conf()->CopyFrom(model_update_conf.naive_conf());
  } else if (model_update_conf.has_momentum_conf()) {
    optimizer_conf->mutable_momentum_conf()->CopyFrom(model_update_conf.momentum_conf());
  } else if (model_update_conf.has_rmsprop_conf()) {
    optimizer_conf->mutable_rmsprop_conf()->CopyFrom(model_update_conf.rmsprop_conf());
  } else if (model_update_conf.has_lars_conf()) {
    optimizer_conf->mutable_lars_conf()->CopyFrom(model_update_conf.lars_conf());
  } else if (model_update_conf.has_adam_conf()) {
    optimizer_conf->mutable_adam_conf()->CopyFrom(model_update_conf.adam_conf());
  } else if (model_update_conf.has_lazy_adam_conf()) {
    optimizer_conf->mutable_lazy_adam_conf()->CopyFrom(model_update_conf.lazy_adam_conf());
  } else if (model_update_conf.has_lamb_conf()) {
    optimizer_conf->mutable_lamb_conf()->CopyFrom(model_update_conf.lamb_conf());
  } else {
    UNIMPLEMENTED();
  }
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("ModelUpdateConfCompatiblePass", ModelUpdateConfCompatiblePass);

}  // namespace

}  // namespace oneflow
