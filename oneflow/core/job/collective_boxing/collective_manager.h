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
#ifndef ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_COLLECTIVE_MANAGER_H_
#define ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_COLLECTIVE_MANAGER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/collective_boxing/runtime_request_info.h"
#include "oneflow/core/job/collective_boxing/of_request_store.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

namespace boxing {

namespace collective {

class RequestHandle;
class CollectiveMgrPlanToken;

class CollectiveMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveMgr);
  ~CollectiveMgr();

  RequestHandle* CreateRequestHandle(const RankDesc& rank_desc);
  void DestroyRequestHandle(RequestHandle*);
  CollectiveMgrPlanToken* AddPlan(const Plan& plan);
  void DeletePlan(CollectiveMgrPlanToken* plan_token);

 private:
  friend class Global<CollectiveMgr>;
  CollectiveMgr();

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_COLLECTIVE_MANAGER_H_
