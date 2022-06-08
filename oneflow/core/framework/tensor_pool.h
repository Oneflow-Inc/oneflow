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

#ifndef ONEFLOW_CORE_FRAMEWORK_TENSOR_POOL_H_
#define ONEFLOW_CORE_FRAMEWORK_TENSOR_POOL_H_

#include <unordered_set>
#include <set>
#include <chrono>

#include "oneflow/core/job/env_global_objects_scope.h"
#include "oneflow/core/eager/dtr_eager_blob_object.h"
#include "oneflow/core/eager/dtr_util.h"

namespace oneflow {
namespace dtr {

struct TensorPool {
  TensorPool();
  OF_DISALLOW_COPY_AND_MOVE(TensorPool);
  ~TensorPool();

  void set_total_memory(size_t mem);
  size_t get_total_memory();

  Maybe<void> insert(std::shared_ptr<vm::DTREagerBlobObject> blob_object, size_t thres = 0);
  Maybe<void> clear();

  void set_current_op_type_name(std::string op_type_name);
  const std::string& current_op_type_name();
  std::string current_op_type_name_;

  Maybe<vm::DTREagerBlobObject*> find_best_tensor();
  Maybe<bool> find_best_tensor_and_evict();
  // do not need Maybe
  const std::chrono::steady_clock::time_point start_time() { return start_time_; }
  double duration();
  void time_flies(double t);
  Maybe<void> display();
  Maybe<void> verbose_display();
  void add_recompute_times() { num_recomputation_++; }
  int recompute_times() const { return num_recomputation_; }
  void merge(std::shared_ptr<vm::DisjNode>& x, std::shared_ptr<vm::DisjNode>& y);
  void pesudo_merge(std::shared_ptr<vm::DisjNode>& x, std::shared_ptr<vm::DisjNode>& y);
  std::shared_ptr<vm::DisjNode> find_father(std::shared_ptr<vm::DisjNode>& x);
  void inc_num_eviction(bool eager_evict);
  void update_after_compute(vm::DTREagerBlobObject* dtr_blob_object);
  Maybe<void> update_after_evict(vm::DTREagerBlobObject* dtr_blob_object);
  int update_after_pesudo_compute(vm::DTREagerBlobObject* dtr_blob_object);
  Maybe<void> update_after_pesudo_evict(vm::DTREagerBlobObject* dtr_blob_object, const char* start_id, const char* end_id);

  std::set<vm::DTRInstrOperand*> operand_visited_;
  std::set<vm::DTREagerBlobObject*> need_eager_eviction_ebos_;

  // TODO: Implementation of disjoint-set data structure

 private:
  double duration_;
  // vector for eviction, set for non-eviction.
  // std::set<std::weak_ptr<vm::DTREagerBlobObject>> candidates_;
  std::vector<std::weak_ptr<vm::DTREagerBlobObject>> candidates_;
  std::chrono::steady_clock::time_point start_time_;
  size_t total_memory_bytes_;   // same in cuda_allocator.h
  int num_eager_eviction_;
  int num_forced_eviction_;
  int num_recomputation_;
  int num_destruction_;
};

}  // namespace dtr
}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_TENSOR_POOL_H_
