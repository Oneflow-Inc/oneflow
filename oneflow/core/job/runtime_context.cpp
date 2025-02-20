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
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

void RuntimeCtx::NewCounter(const std::string& name, int64_t val) {
  VLOG(3) << "NewCounter " << name << " " << val;
  printf("\n ======= RuntimeCtx::NewCounter name:%s %d=======", name.c_str(), int(val));
  CHECK(counters_.emplace(name, std::make_unique<BlockingCounter>(val)).second);
}

void RuntimeCtx::DecreaseCounter(const std::string& name) {
  auto it = counters_.find(name);
  CHECK(it != counters_.end());
  int64_t cur_val = it->second->Decrease();
  printf("\n ======= RuntimeCtx::DecreaseCounter name:%s cnt_val_:%d ======= ", name.c_str(),
         int(cur_val));
  VLOG(3) << "DecreaseCounter " << name << ", current val is " << cur_val;
}

void RuntimeCtx::WaitUntilCntEqualZero(const std::string& name) {
  auto it = counters_.find(name);
  CHECK(it != counters_.end());
  printf("\n ======= RuntimeCtx::WaitUntilCntEqualZero name:%s start ======= ", name.c_str());
  it->second->WaitForeverUntilCntEqualZero();
  printf("\n ======= RuntimeCtx::WaitUntilCntEqualZero name:%s finish ======= ", name.c_str());
  counters_.erase(it);
}

std::string GetRunningActorCountKeyByJobId(int64_t job_id) {
  return "job_" + std::to_string(job_id) + "_running_actor_count";
}

}  // namespace oneflow
