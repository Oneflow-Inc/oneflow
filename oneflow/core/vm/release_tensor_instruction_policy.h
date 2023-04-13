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
#ifndef ONEFLOW_CORE_VM_RELEASE_TENSOR_INSTRUCTION_POLICY_H_
#define ONEFLOW_CORE_VM_RELEASE_TENSOR_INSTRUCTION_POLICY_H_

#include <functional>
#include <memory>
#include "oneflow/core/common/singleton_ptr.h"
#include "oneflow/core/common/throw.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/intrusive/intrusive.h"
#include "oneflow/core/vm/ep_optional_event_record_status_querier.h"
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/eager/tensor_storage.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/vm/stream.h"
#include "oneflow/core/framework/stream_need_soft_sync.h"

namespace oneflow {

namespace vm {

class EagerBlobObject;

class ReleaseTensorInstructionPolicy : public InstructionPolicy {
 public:
  ReleaseTensorInstructionPolicy(const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object,
                                 const Optional<vm::Stream*>& stream)
      : eager_blob_object_(eager_blob_object), output_dependences_() {
    output_dependences_.push_back(CHECK_JUST(eager_blob_object->compute_local_dep_object()));
    if (stream.has_value()) {
      stream_sequential_dependence_ = CHECK_JUST(stream)->schedule_local_dep_object().get();
    }
  }
  ~ReleaseTensorInstructionPolicy() override = default;

  const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object() const {
    return eager_blob_object_;
  }

  const DependenceVector& input_dependences() const override {
    static thread_local DependenceVector empty{};
    return empty;
  }

  const DependenceVector& output_dependences() const override { return output_dependences_; }

  Dependence* stream_sequential_dependence() const override {
    return stream_sequential_dependence_;
  }

 protected:
  void Release(const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object) const {
    CHECK_JUST(eager_blob_object->DeallocateBlobDataPtr());
  }

 private:
  void InitInstructionStatus(Instruction* instruction) override {
    auto* status_buffer = instruction->mut_status_buffer();
    auto* stream = instruction->mut_stream();
    instruction->stream_policy().InitInstructionStatus(*stream, status_buffer);
    auto* data_ptr = status_buffer->mut_buffer();
    EpOptionalEventRecordStatusQuerier::MutCast(data_ptr)->reset_ep_event(nullptr);
  }
  std::shared_ptr<vm::EagerBlobObject> eager_blob_object_;
  DependenceVector output_dependences_;
};

class FastReleaseTensorInstructionPolicy final : public ReleaseTensorInstructionPolicy {
 public:
  using ReleaseTensorInstructionPolicy::ReleaseTensorInstructionPolicy;

  bool Prescheduleable(const vm::Stream* src, const vm::Stream* dst) const override {
    return false;
  }

 private:
  std::string DebugName(const vm::Instruction& instruction) const override {
    return "FastReleaseTensor";
  }

  Maybe<void> Prepare(vm::Instruction* instruction) override {
    DataType data_type = eager_blob_object()->data_type();
    CHECK_OR_RETURN(IsTriviallyCopyableDataType(data_type));
    if (eager_blob_object()->tensor_storage()->is_allocated_in_vm()) {
      Release(eager_blob_object());
    }
    return Maybe<void>::Ok();
  }

  void Compute(vm::Instruction* instruction) override {
    if (!eager_blob_object()->tensor_storage()->is_allocated_in_vm()) {
      Release(eager_blob_object());
    }
  }
};

class SlowReleaseTensorInstructionPolicy final : public ReleaseTensorInstructionPolicy {
 public:
  using ReleaseTensorInstructionPolicy::ReleaseTensorInstructionPolicy;

 private:
  std::string DebugName(const vm::Instruction& instruction) const override {
    return "SlowReleaseTensor";
  }

  Maybe<void> Prepare(vm::Instruction* instruction) override { return Maybe<void>::Ok(); }

  void Compute(vm::Instruction* instruction) override { Release(eager_blob_object()); }
};

struct MakeReleaseTensorInstructionPolicy
    : public StreamTypeVisitor<MakeReleaseTensorInstructionPolicy> {
  static Maybe<vm::InstructionPolicy> VisitCompute(
      const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object,
      const Optional<vm::Stream*>& stream) {
    return Make(eager_blob_object, stream);
  }
  static Maybe<vm::InstructionPolicy> VisitHost2Device(
      const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object,
      const Optional<vm::Stream*>& stream) {
    return Make(eager_blob_object, stream);
  }
  static Maybe<vm::InstructionPolicy> VisitDevice2Host(
      const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object,
      const Optional<vm::Stream*>& stream) {
    return Make(eager_blob_object, stream);
  }
  static Maybe<vm::InstructionPolicy> VisitCcl(
      const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object,
      const Optional<vm::Stream*>& stream) {
    return Make(eager_blob_object, stream);
  }
  static Maybe<vm::InstructionPolicy> VisitBarrier(
      const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object,
      const Optional<vm::Stream*>& stream) {
    UNIMPLEMENTED_THEN_RETURN() << "ReleaseTensor instruction not supported in Barrier stream";
  }
  static Maybe<vm::InstructionPolicy> VisitCriticalSection(
      const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object,
      const Optional<vm::Stream*>& stream) {
    UNIMPLEMENTED_THEN_RETURN()
        << "ReleaseTensor instruction not supported in CriticalSection stream";
  }
  static Maybe<vm::InstructionPolicy> VisitLazyJobLauncher(
      const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object,
      const Optional<vm::Stream*>& stream) {
    UNIMPLEMENTED_THEN_RETURN()
        << "ReleaseTensor instruction not supported in LaunchLazyJob stream";
  }
  static Maybe<vm::InstructionPolicy> VisitPinnedCompute(
      const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object,
      const Optional<vm::Stream*>& stream) {
    return VisitCompute(eager_blob_object, stream);
  }

 private:
  static Maybe<vm::InstructionPolicy> Make(
      const std::shared_ptr<vm::EagerBlobObject>& eager_blob_object,
      const Optional<vm::Stream*>& stream) {
    DataType data_type = eager_blob_object->data_type();
    if (!IsTriviallyCopyableDataType(data_type)) {
      return std::shared_ptr<vm::InstructionPolicy>(
          new vm::SlowReleaseTensorInstructionPolicy(eager_blob_object, stream));
    }
    Symbol<oneflow::Stream> last_used_stream = JUST(eager_blob_object->last_used_stream());
    DeviceType device_type = last_used_stream->device()->enum_type();
    if (NeedSoftSync::Visit(last_used_stream->stream_type(), device_type)) {
      return std::shared_ptr<vm::InstructionPolicy>(
          new vm::SlowReleaseTensorInstructionPolicy(eager_blob_object, stream));
    } else {
      return std::shared_ptr<vm::InstructionPolicy>(
          new vm::FastReleaseTensorInstructionPolicy(eager_blob_object, stream));
    }
  }
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_RELEASE_TENSOR_INSTRUCTION_POLICY_H_
