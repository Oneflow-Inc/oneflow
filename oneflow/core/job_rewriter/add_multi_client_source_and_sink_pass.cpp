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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/job_rewriter/job_pass.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/placement.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/register/logical_blob_id.pb.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {
namespace {

class AddMultiClientSourceAndSinkPass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AddMultiClientSourceAndSinkPass);
  AddMultiClientSourceAndSinkPass() = default;
  ~AddMultiClientSourceAndSinkPass() override = default;

  Maybe<void> Apply(const OpGraph& op_graph, JobBuilder* job_builder) const;

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    if (!JUST(*Global<Maybe<bool>, MultiClient>::Get())) { return Maybe<void>::Ok(); }
    const OpGraph op_graph(*job);
    JobBuilder job_builder(job);
    return Apply(op_graph, &job_builder);
  }
};

Maybe<void> AddMultiClientSourceAndSinkPass::Apply(const OpGraph& op_graph,
                                                   JobBuilder* job_builder) const {
  TODO_THEN_RETURN();
  return Maybe<void>::Ok();
}

}  // namespace

REGISTER_JOB_PASS("AddMultiClientSourceAndSinkPass", AddMultiClientSourceAndSinkPass);

}  // namespace oneflow
