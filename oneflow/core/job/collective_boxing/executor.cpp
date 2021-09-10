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
#include "oneflow/core/job/collective_boxing/executor.h"

namespace oneflow {

namespace boxing {

namespace collective {

void Executor::ExecuteRequests(int64_t job_id, const std::vector<int32_t>& request_ids,
                               void* executor_token) {
  GroupRequests(job_id, request_ids, [&](int64_t job_id, std::vector<int32_t>&& group) {
    ExecuteGroupedRequests(job_id, group, executor_token);
  });
}

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow
