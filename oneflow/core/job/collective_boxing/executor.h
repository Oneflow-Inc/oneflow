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
#ifndef ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_EXECUTOR_H_
#define ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_EXECUTOR_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class CollectiveBoxingPlan;

namespace boxing {

namespace collective {

struct RuntimeRequestInfo;

class RequestStore;

class Executor {
 public:
  Executor() = default;
  virtual ~Executor() = default;

  virtual void Init(std::shared_ptr<RequestStore> request_store) = 0;
  virtual void InitJob(int64_t job_id) = 0;
  virtual void DeinitJob(int64_t job_id) = 0;
  virtual void GroupRequests(
      int64_t job_id, const std::vector<int32_t>& request_ids,
      const std::function<void(int64_t, std::vector<int32_t>&&)>& Handler) = 0;
  virtual void ExecuteGroupedRequests(int64_t job_id, const std::vector<int32_t>& request_ids,
                                      void* executor_token) = 0;

  virtual void ExecuteRequests(int64_t job_id, const std::vector<int32_t>& request_ids,
                               void* executor_token);
  virtual void* CreateExecutorToken(int64_t job_id, int32_t request_id) = 0;
  virtual void DestroyExecutorToken(void* executor_token) = 0;
};

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_EXECUTOR_H_
