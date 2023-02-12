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
#ifndef ONEFLOW_CORE_EAGER_DTR_UTIL_H_
#define ONEFLOW_CORE_EAGER_DTR_UTIL_H_
#include <memory>
#include <vector>

#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/env_var/dtr.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/vm/dtr_ep_allocator.h"

namespace oneflow {

namespace vm {
class DtrEpAllocator;
class RematableTensorStorage;
class OpCallInstructionPolicy;
class DtrOpCallInstructionPolicy;
}

namespace dtr {

bool is_enabled();
size_t memory_threshold();
bool is_enabled_and_debug();
int debug_level();
double append_memory_frag_info_and_get(size_t free_mem, size_t threshold);

class AllocatorManager {
 public:
  vm::DtrEpAllocator* CreateOrGetAllocator(DeviceType device_type, size_t device_index);
 private:
  std::unordered_map<std::pair<DeviceType, size_t>, std::unique_ptr<vm::DtrEpAllocator>> allocators_;
};

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
  vm::OpCallInstructionPolicy update_tensor_with_storage(vm::RematableTensorStorage* storage, const vm::OpCallInstructionPolicy& current_compute_op);

  std::vector<vm::DtrOpCallInstructionPolicy*> ops;

  void add_eviction_num(bool eager_eviction);

  int eager_eviction_num() const { return eager_eviction_num_; }
  int forced_eviction_num() const { return forced_eviction_num_; }

  void add_recomputation_num() { recomputation_num_++; }
  int recomputation_num() const { return recomputation_num_; }

  void clear_time() { time_now_ = 0; }

  std::set<vm::RematableTensorStorage*> need_eager_eviction_storages;

  std::string current_op_type_name;

 private:
  double time_now_ = 0;

  int eager_eviction_num_ = 0;
  int forced_eviction_num_ = 0;
  int recomputation_num_ = 0;

};

Maybe<double> GetComputeTime(const vm::OpCallInstructionPolicy& operand);

}  // namespace dtr

}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_DTR_UTIL_H_
