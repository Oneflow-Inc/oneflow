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
#include "oneflow/core/framework/framework.h"

namespace oneflow {

namespace {

class AddSspVariableProxyPass final : public JobPass {
 public:
  AddSspVariableProxyPass(const AddSspVariableProxyPass&) = delete;
  AddSspVariableProxyPass(AddSspVariableProxyPass&&) = delete;
  AddSspVariableProxyPass() = default;
  ~AddSspVariableProxyPass() = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }

  bool IsEnabled(const JobPassCtx& ctx) const {
    return ctx.job_desc().IsTrain() && ctx.job_desc().Bool("enable_ssp");
  }

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
    TODO();
    return Maybe<void>::Ok();
  }
};

REGISTER_JOB_PASS("AddSspVariableProxy", AddSspVariableProxyPass);

}  // namespace

}  // namespace oneflow
