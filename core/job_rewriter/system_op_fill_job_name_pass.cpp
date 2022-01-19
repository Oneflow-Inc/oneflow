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

class SystemOpFillJobNamePass final : public JobPass {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SystemOpFillJobNamePass);
  SystemOpFillJobNamePass() = default;
  ~SystemOpFillJobNamePass() override = default;

  bool IsEnabled(const JobPassCtx& ctx) const { return true; }

  Maybe<void> Apply(Job* job, JobPassCtx* ctx) const override {
    const std::string& job_name = job->job_conf().job_name();
    for (OperatorConf& op_conf : *job->mutable_net()->mutable_op()) {
      if (op_conf.has_input_conf()) {
        op_conf.mutable_input_conf()->set_job_name(job_name);
      } else if (op_conf.has_wait_and_send_ids_conf()) {
        op_conf.mutable_wait_and_send_ids_conf()->set_job_name(job_name);
      } else if (op_conf.has_output_conf()) {
        op_conf.mutable_output_conf()->set_job_name(job_name);
      } else if (op_conf.has_return_conf()) {
        op_conf.mutable_return_conf()->set_job_name(job_name);
      } else if (op_conf.has_callback_notify_conf()) {
        op_conf.mutable_callback_notify_conf()->set_job_name(job_name);
      } else {
        // do nothing
      }
    }
    return Maybe<void>::Ok();
  }
};

REGISTER_JOB_PASS("SystemOpFillJobNamePass", SystemOpFillJobNamePass);

}  // namespace

}  // namespace oneflow
