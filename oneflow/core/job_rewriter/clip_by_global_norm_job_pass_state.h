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

class ClipByGlobalNormJobPassState : public JobPassState {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ClipByGlobalNormJobPassState);
  ClipByGlobalNormJobPassState() = default;
  ~ClipByGlobalNormJobPassState() override = default;

  class TotalNormState {
   public:
    TotalNormState(const std::string& total_norm_lbn, const std::string& coeff_lbn,
                   const ParallelConf& parallel_conf, int64_t scope_symbol_id)
        : total_norm_lbn_(total_norm_lbn),
          coeff_lbn_(coeff_lbn),
          parallel_conf_(parallel_conf),
          scope_symbol_id_(scope_symbol_id) {}

    void set_total_norm_lbn(const std::string& total_norm_lbn) { total_norm_lbn_ = total_norm_lbn; }
    const std::string& total_norm_lbn() const { return total_norm_lbn_; }
    const std::string& coeff_lbn() const { return coeff_lbn_; }
    const ParallelConf& parallel_conf() const { return parallel_conf_; }
    int64_t scope_symbol_id() const { return scope_symbol_id_; }

   private:
    std::string total_norm_lbn_;
    std::string coeff_lbn_;
    ParallelConf parallel_conf_;
    int64_t scope_symbol_id_;
  };

  void AddTotalNormState(const std::string& variable_op_name,
                         const std::shared_ptr<TotalNormState>& total_norm_state) {
    CHECK(variable_op_name2total_norm_state_.emplace(variable_op_name, total_norm_state).second)
        << variable_op_name;
  }

  const std::shared_ptr<TotalNormState>& GetTotalNormState(const std::string& variable_op_name) {
    const auto& it = variable_op_name2total_norm_state_.find(variable_op_name);
    CHECK(it != variable_op_name2total_norm_state_.end());
    return it->second;
  }

  const bool HasTotalNormState(const std::string& variable_op_name) {
    const auto& it = variable_op_name2total_norm_state_.find(variable_op_name);
    return (it != variable_op_name2total_norm_state_.end());
  }

 private:
  HashMap<std::string, std::shared_ptr<TotalNormState>> variable_op_name2total_norm_state_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_REWRITER_CLIP_BY_GLOBAL_NORM_JOB_PASS_STATE_H_
