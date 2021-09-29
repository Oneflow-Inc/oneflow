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
#ifndef ONEFLOW_CORE_VM_REF_CNT_VM_INSTRUCTION_STATUS_QUERIER_H_
#define ONEFLOW_CORE_VM_REF_CNT_VM_INSTRUCTION_STATUS_QUERIER_H_

#include <atomic>
#include <memory>

namespace oneflow {
namespace vm {

class RefCntInstrStatusQuerier {
 public:
  ~RefCntInstrStatusQuerier() = default;

  bool done() const { return launched_ && *ref_cnt_ == 0; }
  void SetRefCntAndSetLaunched(const std::shared_ptr<std::atomic<int64_t>>& ref_cnt) {
    // No lock needed. This function will be called only one time.
    // In most cases, errors will be successfully detected by CHECK
    // even though run in different threads.
    CHECK(!launched_);
    ref_cnt_ = ref_cnt;
    launched_ = true;
  }

  static const RefCntInstrStatusQuerier* Cast(const char* mem_ptr) {
    return reinterpret_cast<const RefCntInstrStatusQuerier*>(mem_ptr);
  }
  static RefCntInstrStatusQuerier* MutCast(char* mem_ptr) {
    return reinterpret_cast<RefCntInstrStatusQuerier*>(mem_ptr);
  }
  static RefCntInstrStatusQuerier* PlacementNew(char* mem_ptr) {
    return new (mem_ptr) RefCntInstrStatusQuerier();
  }

 private:
  RefCntInstrStatusQuerier() : launched_(false), ref_cnt_() {}

  std::atomic<bool> launched_;
  std::shared_ptr<std::atomic<int64_t>> ref_cnt_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_REF_CNT_VM_INSTRUCTION_STATUS_QUERIER_H_
