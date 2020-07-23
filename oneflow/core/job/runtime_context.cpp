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
  LOG(INFO) << "NewCounter " << name << " " << val;
  CHECK(counters_.emplace(name, std::make_unique<BlockingCounter>(val)).second);
}

void RuntimeCtx::DecreaseCounter(const std::string& name) {
  int64_t cur_val = counters_.at(name)->Decrease();
  LOG(INFO) << "DecreaseCounter " << name << ", current val is " << cur_val;
}

void RuntimeCtx::WaitUntilCntEqualZero(const std::string& name) {
  counters_.at(name)->WaitUntilCntEqualZero();
}

RuntimeCtx::RuntimeCtx(int64_t total_piece_num, bool is_experiment_phase) {
  total_piece_num_ = total_piece_num;
  is_experiment_phase_ = is_experiment_phase;
}

}  // namespace oneflow
