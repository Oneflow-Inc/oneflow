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
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/job/job_set_compile_ctx.h"

namespace oneflow {

namespace {

class SetDefaultVariableConf final : public JobPass {
 public:
  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
    op_graph.ForEachNode([&](OpNode* op_node) {
      if (op_node->op().op_conf().has_variable_conf()) {
        OperatorConf variable_op_conf(op_node->op().op_conf());
        VariableOpConf* variable_conf = variable_op_conf.mutable_variable_conf();
        if (!variable_conf->has_data_type()) {
          variable_conf->set_data_type(job_builder->job().job_conf().default_data_type());
        }
        if (!variable_conf->has_initializer() && !variable_conf->has_initialize_with_snapshot()) {
          if (job_builder->job().job_conf().has_default_initializer_conf()) {
            *variable_conf->mutable_initializer() =
                job_builder->job().job_conf().default_initializer_conf();
          } else if (job_builder->job().job_conf().has_default_initialize_with_snapshot_path()) {
            variable_conf->mutable_initialize_with_snapshot()->set_path(
                job_builder->job().job_conf().default_initialize_with_snapshot_path());
            variable_conf->mutable_initialize_with_snapshot()->set_key(
                GenLogicalBlobName(op_node->op().BnInOp2Lbi("out")));
          } else {
            UNIMPLEMENTED();
          }
        }
        int64_t random_seed;
        auto* var_op_name2random = Global<JobSetCompileCtx>::Get()->GetVarOpName2randomSeed();
        const std::string& var_op_name = variable_op_conf.name();
        if (variable_conf->has_random_seed()) {
          random_seed = variable_conf->random_seed();
        } else {
          random_seed = NewRandomSeed();
        }
        const auto& pair = var_op_name2random->insert({var_op_name, random_seed});
        if (variable_conf->has_random_seed()) {
          CHECK_EQ(variable_conf->random_seed(), pair.first->second);
        } else {
          variable_conf->set_random_seed(pair.first->second);
        }
        job_builder->AddOrMutOpsOnlyOnce(op_node->parallel_desc().parallel_conf(),
                                         {variable_op_conf});
      }
    });
    return Maybe<void>::Ok();
  }
};

REGISTER_JOB_PASS("SetDefaultVariableConf", SetDefaultVariableConf);

}  // namespace

}  // namespace oneflow
