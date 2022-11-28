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
#ifndef ONEFLOW_CORE_VM_ACCESS_BLOB_ARG_CB_INSTRUCTION_POLICY_H_
#define ONEFLOW_CORE_VM_ACCESS_BLOB_ARG_CB_INSTRUCTION_POLICY_H_

#include <functional>
#include <memory>
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/instruction_policy.h"
#include "oneflow/core/vm/instruction_policy_util.h"
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/tensor_storage.h"
#include "oneflow/core/intrusive/list.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/vm/stream_policy.h"

namespace oneflow {
namespace vm {

class AccessBlobArgCbInstructionPolicy final : public InstructionPolicy {
 public:
  AccessBlobArgCbInstructionPolicy(
      const std::shared_ptr<EagerBlobObject>& eager_blob_object,
      const std::function<void(ep::Stream*, const std::shared_ptr<vm::EagerBlobObject>&)>& callback,
      const std::string& modifier)
      : eager_blob_object_(eager_blob_object),
        callback_(callback),
        modifier_(modifier),
        input_dependences_(),
        output_dependences_() {
    ForEachConstDependence(InstructionPolicyUtil::SetInserter(&input_dependences_));
    ForEachMutDependence(InstructionPolicyUtil::SetInserter(&output_dependences_));
    ForEachMut2Dependence(InstructionPolicyUtil::SetInserter(&output_dependences_));
    stream_sequential_dependence_ = nullptr;
  }
  ~AccessBlobArgCbInstructionPolicy() = default;

  const std::shared_ptr<EagerBlobObject>& eager_blob_object() const { return eager_blob_object_; }

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

  void ForEachConstDependence(const std::function<void(Dependence* compute)>& DoEach) const {
    if (modifier_ == "const") {
      DoEach(CHECK_JUST(eager_blob_object_->compute_local_dep_object()));
    }
  }

  void ForEachMutDependence(const std::function<void(Dependence* compute)>& DoEach) const {
    if (modifier_ == "mut") { DoEach(CHECK_JUST(eager_blob_object_->compute_local_dep_object())); }
  }

  void ForEachMut2Dependence(const std::function<void(Dependence* compute)>& DoEach) const {
    if (modifier_ == "mut2") { DoEach(CHECK_JUST(eager_blob_object_->compute_local_dep_object())); }
  }

  std::string DebugName(const Instruction& instruction) const override {
    return "AccessBlobByCallback";
  }
  Maybe<void> Prepare(Instruction* instruction) override { return Maybe<void>::Ok(); }
  void Compute(Instruction* instruction) override {
    StreamPolicy* stream_policy = instruction->mut_stream_policy();
    return callback_(stream_policy->stream(), eager_blob_object());
  }

 private:
  std::shared_ptr<EagerBlobObject> eager_blob_object_;
  std::function<void(ep::Stream*, const std::shared_ptr<vm::EagerBlobObject>&)> callback_;
  const std::string modifier_;
  DependenceVector input_dependences_;
  DependenceVector output_dependences_;
};

}  // namespace vm
}  // namespace oneflow
#endif  // ONEFLOW_CORE_VM_ACCESS_BLOB_ARG_CB_INSTRUCTION_POLICY_H_
