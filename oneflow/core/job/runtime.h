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
#ifndef ONEFLOW_CORE_JOB_RUNTIME_H_
#define ONEFLOW_CORE_JOB_RUNTIME_H_

#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/job/collective_boxing/scheduler.h"

namespace oneflow {

namespace vm {
class EagerBlobObject;
}

class Runtime final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Runtime);
  Runtime() = delete;
  ~Runtime();

  // TODO(chengcheng): refactor Runtime interface about variable_op_name2eager_blob_object
  Runtime(const Plan& plan,
          const HashMap<std::string, vm::EagerBlobObject*>& variable_op_name2eager_blob_object);

 private:
  void DumpThreadIdsFromPlan(const Plan& plan);

  HashMap<int64_t, int64_t> job_id2actor_size_;
  HashSet<int64_t> thread_ids_;
  HashSet<int64_t> independent_thread_ids_;

  boxing::collective::SchedulerPlanToken* collective_boxing_scheduler_plan_token_;
};

}  // namespace oneflow

#endif
