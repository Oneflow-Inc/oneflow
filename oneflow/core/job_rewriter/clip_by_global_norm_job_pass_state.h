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

class ClipByGlobalNormToken {
 public:
  ClipByGlobalNormToken(const int index, const std::string& total_norm_lbn,
                        const std::string coeff_lbn, const ParallelConf& parallel_conf,
                        int64_t scope_symbol_id)
      : index_(index),
        total_norm_lbn_(total_norm_lbn),
        coeff_lbn_(coeff_lbn),
        parallel_conf_(parallel_conf),
        scope_symbol_id_(scope_symbol_id) {}

  void set_total_norm_lbn(const std::string& total_norm_lbn) { total_norm_lbn_ = total_norm_lbn; }
  int index() const { return index_; }
  const std::string& total_norm_lbn() const { return total_norm_lbn_; }
  const std::string& coeff_lbn() const { return coeff_lbn_; }
  const ParallelConf& parallel_conf() const { return parallel_conf_; }
  int64_t scope_symbol_id() const { return scope_symbol_id_; }

  bool operator==(const ClipByGlobalNormToken& other) const {
    if (index_ != other.index()) { return false; }
    if (total_norm_lbn_ != other.total_norm_lbn()) { return false; }
    if (coeff_lbn_ != other.coeff_lbn()) { return false; }
    if (parallel_conf_ != other.parallel_conf()) { return false; }
    if (scope_symbol_id_ != other.scope_symbol_id()) { return false; }
    return true;
  }

 private:
  int index_;
  std::string total_norm_lbn_;
  std::string coeff_lbn_;
  ParallelConf parallel_conf_;
  int64_t scope_symbol_id_;
};

class ClipByGlobalNormJobPassState : public JobPassState {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ClipByGlobalNormJobPassState);
  ClipByGlobalNormJobPassState() = default;
  ~ClipByGlobalNormJobPassState() override = default;

  const ClipByGlobalNormToken& CreateClipByGlobalNormToken(const std::string& total_norm_lbn,
                                                           const std::string coeff_lbn,
                                                           const ParallelConf& parallel_conf,
                                                           int64_t scope_symbol_id) {
    int index = tokens_.size();
    ClipByGlobalNormToken token(index, total_norm_lbn, coeff_lbn, parallel_conf, scope_symbol_id);
    tokens_.push_back(token);
    return tokens_.at(index);
  }

  void SetClipByGlobalNormToken(const std::string& variable_op_name,
                                const ClipByGlobalNormToken& token) {
    const auto& it = variable_op_name2token_index_.find(variable_op_name);
    if (it != variable_op_name2token_index_.end()) {
      int index = it->second;
      if (token == tokens_.at(index)) {
        // do nothing
      } else {
        CHECK_EQ(token.index(), index) << "token's index can't be changed";
        tokens_.at(index) = token;
      }
    } else {
      CHECK(tokens_.at(token.index()) == token);
      CHECK(variable_op_name2token_index_.emplace(variable_op_name, token.index()).second)
          << "variable_op_name " << variable_op_name;
    }
  }

  const ClipByGlobalNormToken& GetClipByGlobalNormToken(const std::string& variable_op_name) {
    const auto& it = variable_op_name2token_index_.find(variable_op_name);
    CHECK(it != variable_op_name2token_index_.end());
    int index = it->second;
    return tokens_.at(index);
  }

  const bool HasClipByGlobalNormToken(const std::string& variable_op_name) {
    const auto& it = variable_op_name2token_index_.find(variable_op_name);
    return (it != variable_op_name2token_index_.end());
  }

 private:
  std::vector<ClipByGlobalNormToken> tokens_;
  HashMap<std::string, int> variable_op_name2token_index_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_REWRITER_CLIP_BY_GLOBAL_NORM_JOB_PASS_STATE_H_
