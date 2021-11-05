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
#include "oneflow/core/auto_parallel/sbp_constructor.h"

namespace oneflow {

namespace {

class AutoParallelPass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AutoParallelPass);
  AutoParallelPass() = default;
  ~AutoParallelPass() override = default;

  Maybe<void> Apply(const OpGraph& op_graph, Job* job) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!job->job_conf().enable_auto_parallel()) { return Maybe<void>::Ok(); }
    LOG(INFO) << "=== Enable AutoParallel ===";
    const OpGraph op_graph(*job);
    return Apply(op_graph, job);
  }
};

Maybe<void> AutoParallelPass::Apply(const OpGraph& op_graph, Job* job) const {
  auto_parallel::SbpConstructor sbp_constructor(op_graph, job);
  sbp_constructor.FindBestSbpSignature();
  sbp_constructor.UpdateSbpSignatureForJob(op_graph);
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("AutoParallelPass", AutoParallelPass);

}  // namespace

}  // namespace oneflow
