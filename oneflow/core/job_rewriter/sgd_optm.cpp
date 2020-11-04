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
#include "oneflow/core/job_rewriter/optimizer.h"
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

void GenerateOptimizerOpConf(JobPassCtx* ctx, const VariableOp& op,
                             const ParallelConf& parallel_conf, JobBuilder* job_builder,
                             const LogicalBlobId& diff_lbi_of_var_out) {
  const auto& train_conf = job_builder->job().job_conf().train_conf();
  const NormalModelUpdateOpUserConf& model_update_conf = train_conf.model_update_conf();
  user_op::UserOpConfWrapperBuilder sgd_update_op_builder(op.op_name() + "_optimizer");
  sgd_update_op_builder.OpTypeName("sgd_update")
      .Input("model", GenLogicalBlobName(op.BnInOp2Lbi("out")))
      .Input("model_diff", GenLogicalBlobName(diff_lbi_of_var_out))
      .Input("learning_rate", train_conf.primary_lr_lbn())
      .Attr<float>("weight_decay", GetOptimizerWeightDecayRate(model_update_conf, op))
      .ScopeSymbolId(op.op_conf().scope_symbol_id());
  user_op::UserOpConfWrapper sgd_update_op = sgd_update_op_builder.Build();
  job_builder->AddOps(parallel_conf, {sgd_update_op.op_conf()});
}

}  // namespace

REGISTER_OPTIMIZER(NormalModelUpdateOpUserConf::kNaiveConf, &GenerateOptimizerOpConf);

}  // namespace oneflow
