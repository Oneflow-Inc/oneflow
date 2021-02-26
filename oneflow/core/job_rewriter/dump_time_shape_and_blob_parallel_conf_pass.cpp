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

namespace oneflow {

namespace {

class DumpTimeShapeAndBlobParallelConfPass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DumpTimeShapeAndBlobParallelConfPass);
  DumpTimeShapeAndBlobParallelConfPass() = default;
  ~DumpTimeShapeAndBlobParallelConfPass() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const { return true; }

  Maybe<void> Apply(const OpGraph& op_graph, Job* job) const {
    op_graph.DumpOpTimeShape(job);
    op_graph.DumpLogicalBlobDesc(job);
    op_graph.DumpSbpSignature(job);
    return Maybe<void>::Ok();
  }

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    const OpGraph op_graph(*job);
    return Apply(op_graph, job);
  }
};

REGISTER_JOB_PASS("DumpTimeShapeAndBlobParallelConfPass", DumpTimeShapeAndBlobParallelConfPass);

}  // namespace

}  // namespace oneflow
