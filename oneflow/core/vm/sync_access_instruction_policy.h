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
#ifndef ONEFLOW_CORE_VM_SYNC_ACCESS_INSTRUCTION_POLICY_H_
#define ONEFLOW_CORE_VM_SYNC_ACCESS_INSTRUCTION_POLICY_H_

#include <functional>
#include <memory>
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/instruction_policy.h"
#include "oneflow/core/vm/instruction_policy_util.h"
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/tensor_storage.h"
#include "oneflow/core/common/blocking_then_busy.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/stream_policy.h"
#include "oneflow/core/memory/memory_case_util.h"

namespace oneflow {
namespace vm {

class SyncAccessInstructionPolicy : public InstructionPolicy {
 public:
  SyncAccessInstructionPolicy();
  virtual ~SyncAccessInstructionPolicy() = default;

  Maybe<void> Prepare(Instruction* instruction) override { return Maybe<void>::Ok(); }

  BlockingThenBusy* mut_btb() { return &btb_; }

 protected:
  void ResetBase(char* mem_ptr, size_t bytes, EagerBlobObject* eager_blob_object);

  const MemoryCase host_mem_case_;
  BlockingThenBusy btb_;
  char* mem_ptr_;
  size_t bytes_;
  EagerBlobObject* eager_blob_object_;
};

class SyncReadInstructionPolicy final : public SyncAccessInstructionPolicy {
 public:
  SyncReadInstructionPolicy() = default;
  ~SyncReadInstructionPolicy() = default;

  const DependenceVector& input_dependences() const override {
    CHECK_EQ(input_dependences_.size(), 1);
    return input_dependences_;
  }

  const DependenceVector& output_dependences() const override {
    static thread_local DependenceVector empty{};
    return empty;
  }

  std::string DebugName(const Instruction& instruction) const override { return "SyncRead"; }

  void Reset(char* mem_ptr, size_t bytes, EagerBlobObject* eager_blob_object) {
    ResetBase(mem_ptr, bytes, eager_blob_object);
    if (likely(input_dependences_.size())) { input_dependences_.clear(); }
    input_dependences_.push_back(CHECK_JUST(eager_blob_object->compute_local_dep_object()));
  }

  void Compute(Instruction* instruction) override;

 private:
  DependenceVector input_dependences_;
};

}  // namespace vm
}  // namespace oneflow
#endif  // ONEFLOW_CORE_VM_SYNC_ACCESS_INSTRUCTION_POLICY_H_
