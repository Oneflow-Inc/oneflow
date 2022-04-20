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
#ifndef ONEFLOW_CORE_JOB_REWRITER_CLIP_BY_GLOBAL_NORM_JOB_PASS_STATE_H_
#define ONEFLOW_CORE_JOB_REWRITER_CLIP_BY_GLOBAL_NORM_JOB_PASS_STATE_H_

#include "oneflow/core/job_rewriter/job_pass.h"

namespace oneflow {

struct ClipByGlobalNormState {
  std::string total_norm_lbn;
  std::string coeff_lbn;
  ParallelConf parallel_conf;
  ClipByGlobalNormState(const std::string& total_norm_lbn, const std::string coeff_lbn, const ParallelConf& parallel_conf):total_norm_lbn(total_norm_lbn), coeff_lbn(coeff_lbn), parallel_conf(parallel_conf) {}
}


class ClipByGlobalNormJobPassState : public JobPassState {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ClipByGlobalNormJobPassState);
  ClipByGlobalNormJobPassState() = default;
  ~ClipByGlobalNormJobPassState() override = default;

  const ClipByGlobalNormState& clip_by_global_norm_state(const OptimizerConf& optimizer_conf) const { 
    const auto& it = optimizer_conf2clip_by_global_norm_state_.find(optimizer_conf);
    CHECK(it != optimizer_conf2clip_by_global_norm_state_.end())<<"current optimizer has no clip_by_global_norm_state";
    return it->second; 
  }
  void set_clip_by_global_norm_state(const OptimizerConf& optimizer_conf, const ClipByGlobalNormState& state) { 
    const auto& it = optimizer_conf2clip_by_global_norm_state_.find(optimizer_conf);
    if(it == optimizer_conf2clip_by_global_norm_state_.end()) {
      CHECK(optimizer_conf2clip_by_global_norm_state_.emplace(optimizer_conf, state).second);
    } else {
      it->second = state;
    }
  }

 private:
  HashMap<OptimizerConf, ClipByGlobalNormState> optimizer_conf2clip_by_global_norm_state_;

};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_REWRITER_CLIP_BY_GLOBAL_NORM_JOB_PASS_STATE_H_
