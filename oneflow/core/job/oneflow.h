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
#ifndef ONEFLOW_CORE_JOB_ONEFLOW_H_
#define ONEFLOW_CORE_JOB_ONEFLOW_H_

#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/rpc/include/local/ctrl.h"
#include "oneflow/core/job/runtime.h"
#include "oneflow/core/job/runtime_buffers_scope.h"
#include "oneflow/core/job/inter_user_job_info.pb.h"

namespace oneflow {

class Oneflow final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Oneflow);
  Oneflow() {}
  ~Oneflow();

  Maybe<void> Init(const oneflow::JobSet& job_set);

 private:
  Plan plan_;
  std::unique_ptr<RuntimeBuffersScope> runtime_buffers_scope_;
  std::unique_ptr<Runtime> runtime_;
};

int Main(const oneflow::JobSet& job_set);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_ONEFLOW_H_
