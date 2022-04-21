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

namespace std {

template<>
struct hash<oneflow::PbRpf<std::string>> {
  size_t operator()(const oneflow::PbRpf<std::string>& fields) const {
    const auto& str_hash = std::hash<std::string>();
    size_t hash = 0;
    for (int i = 0; i < fields.size(); ++i) { oneflow::HashCombine(&hash, str_hash(fields.at(i))); }
    return hash;
  }
};

template<>
struct hash<oneflow::OptimizerConf> {
  size_t operator()(const oneflow::OptimizerConf& conf) const {
    return std::hash<oneflow::PbRpf<std::string>>()(conf.variable_op_names());
  }
};

}  // namespace std

namespace oneflow {

struct ClipByGlobalNormState {
  std::string total_norm_lbn;
  std::string coeff_lbn;
  ParallelConf parallel_conf;
  int64_t scope_symbol_id;
  ClipByGlobalNormState(const std::string& total_norm_lbn, const std::string coeff_lbn,
                        const ParallelConf& parallel_conf, int64_t scope_symbol_id)
      : total_norm_lbn(total_norm_lbn),
        coeff_lbn(coeff_lbn),
        parallel_conf(parallel_conf),
        scope_symbol_id(scope_symbol_id) {}
  ClipByGlobalNormState& operator=(const ClipByGlobalNormState& other) {
    total_norm_lbn = other.total_norm_lbn;
    coeff_lbn = other.coeff_lbn;
    parallel_conf = other.parallel_conf;
    scope_symbol_id = other.scope_symbol_id;
    return *this;
  }
};

inline bool operator==(const oneflow::PbRpf<std::string>& lhs,
                       const oneflow::PbRpf<std::string>& rhs) {
  if (lhs.size() != rhs.size()) { return false; }
  for (int i = 0; i < lhs.size(); ++i) {
    if (lhs.at(i) != rhs.at(i)) { return false; }
  }
  return true;
}

inline bool operator==(const OptimizerConf& lhs, const OptimizerConf& rhs) {
  return lhs.variable_op_names() == rhs.variable_op_names();
}

class ClipByGlobalNormJobPassState : public JobPassState {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ClipByGlobalNormJobPassState);
  ClipByGlobalNormJobPassState() = default;
  ~ClipByGlobalNormJobPassState() override = default;

  const ClipByGlobalNormState& clip_by_global_norm_state(
      const OptimizerConf& optimizer_conf) const {
    const auto& it = optimizer_conf2clip_by_global_norm_state_.find(optimizer_conf);
    CHECK(it != optimizer_conf2clip_by_global_norm_state_.end())
        << "current optimizer has no clip_by_global_norm_state";
    return it->second;
  }
  void set_clip_by_global_norm_state(const OptimizerConf& optimizer_conf,
                                     const ClipByGlobalNormState& state) {
    const auto& it = optimizer_conf2clip_by_global_norm_state_.find(optimizer_conf);
    if (it == optimizer_conf2clip_by_global_norm_state_.end()) {
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
