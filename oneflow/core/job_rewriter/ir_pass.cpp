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
#include "oneflow/ir/oneflow-translate/include/OneFlow/MLIROneFlowTranslation.h"

namespace oneflow {

namespace {

class IRRoundTrip final : public JobPass {
 public:
  IRRoundTrip() = default;
  ~IRRoundTrip() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const {
    // TODO: add compiler definition for MLIR and job conf flags
    return true;
  }
  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!IsEnabled(*ctx)) { return Maybe<void>::Ok(); }
    mlir::roundTripOneFlowJob(job);
    return Maybe<void>::Ok();
  }
};

}  // namespace

REGISTER_JOB_PASS("IRRoundTrip", IRRoundTrip);

}  // namespace oneflow
