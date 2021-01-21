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
#ifndef ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_DYNAMIC_COORDINATOR_H_
#define ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_DYNAMIC_COORDINATOR_H_

#include "oneflow/core/job/collective_boxing/coordinator.h"

#ifdef WITH_MPI

namespace oneflow {

class CollectiveBoxingPlan;

namespace boxing {

namespace collective {

class RequestStore;
class Executor;

class DynamicCoordinator : public Coordinator {
 public:
  OF_DISALLOW_COPY_AND_MOVE(DynamicCoordinator);
  DynamicCoordinator();
  ~DynamicCoordinator() override;

  void Init(const CollectiveBoxingPlan& collective_boxing_plan,
            std::shared_ptr<RequestStore> request_store,
            std::shared_ptr<Executor> executor) override;
  void AddRequest(int32_t request_id) override;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow

#endif  // WITH_MPI

#endif  // ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_DYNAMIC_COORDINATOR_H_
