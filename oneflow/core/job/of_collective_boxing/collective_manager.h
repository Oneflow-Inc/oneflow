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
#ifndef ONEFLOW_CORE_JOB_OF_COLLECTIVE_BOXING_COLLECTIVE_MANAGER_H_
#define ONEFLOW_CORE_JOB_OF_COLLECTIVE_BOXING_COLLECTIVE_MANAGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/of_collective_boxing/of_request_store.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

namespace boxing {

namespace of_collective {

class CollectiveMgrPlanToken;

class CollectiveMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveMgr);
  ~CollectiveMgr();

  CollectiveMgrPlanToken* AddPlan(const Plan& plan);
  void DeletePlan(CollectiveMgrPlanToken* plan_token);

  OfRequestId GetOfRequestIdByName(const std::string& name) const;

  void* CreateOfRequestEntryToken(const OfRequestId& request_id);

  void DestroyOfRequestEntryToken(void* token);

  OfRequestEntry* GetOfRequestEntry(void* token);

 private:
  friend class Singleton<CollectiveMgr>;
  CollectiveMgr();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace of_collective

}  // namespace boxing

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_OF_COLLECTIVE_BOXING_COLLECTIVE_MANAGER_H_
