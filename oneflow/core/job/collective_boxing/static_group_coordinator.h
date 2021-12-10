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
#ifndef ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_STATIC_GROUP_COORDINATOR_H_
#define ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_STATIC_GROUP_COORDINATOR_H_

#include "oneflow/core/job/collective_boxing/coordinator.h"

namespace oneflow {

namespace boxing {

namespace collective {

class RequestStore;
class Executor;

class StaticGroupCoordinator : public Coordinator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(StaticGroupCoordinator);
  StaticGroupCoordinator();
  ~StaticGroupCoordinator() override;

  void Init(std::shared_ptr<RequestStore> request_store,
            std::shared_ptr<Executor> executor) override;
  void InitJob(int64_t job_id) override;
  void DeinitJob(int64_t job_id) override;
  void AddRequest(void* coordinator_token) override;
  void* CreateCoordinatorToken(const RequestId& request_id) override;
  void DestroyCoordinatorToken(void* token) override;

 private:
  void DumpSummary(const int64_t job_id) const;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_STATIC_GROUP_COORDINATOR_H_
