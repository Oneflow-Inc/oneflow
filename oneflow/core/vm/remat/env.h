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
#pragma once

#include "oneflow/core/common/env_var/remat.h"
#include "oneflow/core/common/util.h"

#define VLOG_REMAT(verbose_level) \
  if (Singleton<remat::Env>::Get()->log_enabled()) VLOG(verbose_level)

namespace oneflow {

namespace vm {
class RematableTensorStorage;
class OpCallInstructionPolicy;
class DtrOpCallInstructionPolicy;
}  // namespace vm

namespace remat {

class Env {
 public:
  Env() = default;
  ~Env();
  OF_DISALLOW_COPY_AND_MOVE(Env);
  double time_now() { return time_now_; }
  void add_time(double time) { time_now_ += time; }
  void remove_compute_op(vm::DtrOpCallInstructionPolicy* op) {
    ops.erase(std::remove(ops.begin(), ops.end(), op), ops.end());
  }
  vm::OpCallInstructionPolicy update_tensor_with_storage(
      vm::RematableTensorStorage* storage, const vm::OpCallInstructionPolicy& current_compute_op);

  std::vector<vm::DtrOpCallInstructionPolicy*> ops;

  void add_eviction_num(bool eager_eviction);

  int eager_eviction_num() const { return eager_eviction_num_; }
  int forced_eviction_num() const { return forced_eviction_num_; }

  void add_recomputation_num() { recomputation_num_++; }
  int recomputation_num() const { return recomputation_num_; }

  void clear_stats() {
    time_now_ = 0;
    eager_eviction_num_ = 0;
    forced_eviction_num_ = 0;
    recomputation_num_ = 0;
  }

  std::set<vm::RematableTensorStorage*> need_eager_eviction_storages;

  void set_budget_in_bytes(size_t budget_in_bytes) { budget_in_bytes_ = budget_in_bytes; }
  size_t budget_in_bytes() const { return budget_in_bytes_; }

  void set_small_pieces_optimization(bool enabled) { small_pieces_optimization_ = enabled; }
  bool is_small_pieces_optimization_enabled() const { return small_pieces_optimization_; }

  bool log_enabled() const { return EnvBool<ONEFLOW_REMAT_LOG>(); }

 private:
  double time_now_ = 0;

  int eager_eviction_num_ = 0;
  int forced_eviction_num_ = 0;
  int recomputation_num_ = 0;

  size_t budget_in_bytes_ = 0;
  bool small_pieces_optimization_ = true;
};

struct CurrentOpTypeName {
  std::string value;
};

}  // namespace remat
}  // namespace oneflow
