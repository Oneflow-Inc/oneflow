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
#ifndef ONEFLOW_CORE_VM_NAIVE_VM_INSTRUCTION_STATUS_QUERIER_H_
#define ONEFLOW_CORE_VM_NAIVE_VM_INSTRUCTION_STATUS_QUERIER_H_

#include <atomic>

namespace oneflow {
namespace vm {

class NaiveInstrStatusQuerier {
 public:
  ~NaiveInstrStatusQuerier() = default;

  bool launched() const { return done_; }
  bool done() const { return done_; }
  void set_done() { done_ = true; }

  static const NaiveInstrStatusQuerier* Cast(const char* mem_ptr) {
    return reinterpret_cast<const NaiveInstrStatusQuerier*>(mem_ptr);
  }
  static NaiveInstrStatusQuerier* MutCast(char* mem_ptr) {
    return reinterpret_cast<NaiveInstrStatusQuerier*>(mem_ptr);
  }
  static NaiveInstrStatusQuerier* PlacementNew(char* mem_ptr) {
    return new (mem_ptr) NaiveInstrStatusQuerier();
  }

 private:
  NaiveInstrStatusQuerier() : done_(false) {}
  std::atomic<bool> done_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_NAIVE_VM_INSTRUCTION_STATUS_QUERIER_H_
