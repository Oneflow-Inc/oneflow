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

#include "oneflow/core/common/just.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/job_builder.h"
#include "oneflow/core/job_rewriter/job_pass.h"
namespace oneflow {

namespace {

class StraightenOpGraphPass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StraightenOpGraphPass);
  StraightenOpGraphPass() = default;
  ~StraightenOpGraphPass() = default;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;
};

Maybe<void> StraightenOpGraphPass::Apply(const OpGraph& op_graph, JobBuilder* job_builder) const {
  std::cout << "Straighten op graph is working!" << std::endl;
  return Maybe<void>::Ok();
}

}  // anonymous namespace

REGISTER_JOB_PASS("StraightenOpGraphPass", StraightenOpGraphPass);

}  // namespace oneflow