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
#ifndef ONEFLOW_CORE_JOB_RUNTIME_CONTEXT_H_
#define ONEFLOW_CORE_JOB_RUNTIME_CONTEXT_H_

#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/job_set.pb.h"
#include "oneflow/core/persistence/persistent_in_stream.h"
#include "oneflow/core/persistence/persistent_out_stream.h"

namespace oneflow {

class RuntimeCtx final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RuntimeCtx);
  RuntimeCtx() = delete;
  ~RuntimeCtx() = default;

  int64_t total_piece_num() const { return total_piece_num_; }
  bool is_experiment_phase() const { return is_experiment_phase_; }
  bool NeedCollectActEvent() const {
    return is_experiment_phase_ || Global<const ProfilerConf>::Get()->collect_act_event();
  }

  void NewCounter(const std::string& name, int64_t val);
  void DecreaseCounter(const std::string& name);
  void WaitUntilCntEqualZero(const std::string& name);

 private:
  friend class Global<RuntimeCtx>;
  RuntimeCtx(int64_t total_piece_num, bool is_experiment_phase);

  int64_t total_piece_num_;
  bool is_experiment_phase_;
  HashMap<std::string, std::unique_ptr<BlockingCounter>> counters_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_RUNTIME_CONTEXT_H_
