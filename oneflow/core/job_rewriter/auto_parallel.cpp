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
#include <chrono>
#include "oneflow/core/common/util.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/auto_parallel/sbp_constructor.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"

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
  // auto-parallel
  // TODO: recode this
  std::cout << "Start Auto Parallel" << std::endl;
  auto time_begin = std::chrono::high_resolution_clock::now();

  auto_parallel::SbpConstructor sbp_constructor(op_graph, job);
  JUST(sbp_constructor.FindBestSbpSignature());
  JUST(sbp_constructor.DumpNdSbpSignatureForJob(op_graph, job));
  auto time_end = std::chrono::high_resolution_clock::now();
  std::cout << "Auto parallel took "
            << std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_begin).count()
            << " ms\n";
  if (GlobalProcessCtx::Rank() == 0) {
    sbp_constructor.PrintSBPGraphDebugInfo();
    JUST(sbp_constructor.CheckSbpAgreement(*job));
  }
  return Maybe<void>::Ok();
}

REGISTER_JOB_PASS("AutoParallelPass", AutoParallelPass);

}  // namespace

}  // namespace oneflow
