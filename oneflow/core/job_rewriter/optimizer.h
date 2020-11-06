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
#ifndef ONEFLOW_CORE_JOB_REWRITER_OPTIMIZER_H_
#define ONEFLOW_CORE_JOB_REWRITER_OPTIMIZER_H_

#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/operator/variable_op.h"
#include "oneflow/core/job_rewriter/job_pass.h"

namespace oneflow {

void AddOptimizerOpConf(JobPassCtx* ctx, const OpGraph& op_graph, JobBuilder* job_builder,
                        const HashMap<LogicalBlobId, LogicalBlobId>& lbi2diff_lbi);

float GetOptimizerWeightDecayRate(const NormalModelUpdateOpUserConf& model_update_conf,
                                  const VariableOp& op);

template<typename T>
void ConstructMdUpdtOpConf(const VariableOp& op, const LogicalBlobId& diff_lbi_of_var_out,
                           JobBuilder* job_builder, T*);

class GenerateOptimizerOpConfWrapperStruct final {
 public:
  using Func = std::function<void(JobPassCtx*, const VariableOp&, const ParallelConf&, JobBuilder*,
                                  const LogicalBlobId&)>;
  GenerateOptimizerOpConfWrapperStruct(const Func& f) : func_(std::make_unique<Func>(f)) {}
  void Call(JobPassCtx* ctx, const VariableOp& op, const ParallelConf& parallel_conf,
            JobBuilder* job_builder, const LogicalBlobId& diff_lbi_of_var_out) const;

 private:
  const std::unique_ptr<const Func> func_;
};

#define REGISTER_OPTIMIZER(model_update_case, gen_grad_func)                               \
  REGISTER_CLASS_CREATOR(int32_t, model_update_case, GenerateOptimizerOpConfWrapperStruct, \
                         ([] { return new GenerateOptimizerOpConfWrapperStruct(gen_grad_func); }))

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_REWRITER_OPTIMIZER_H_
