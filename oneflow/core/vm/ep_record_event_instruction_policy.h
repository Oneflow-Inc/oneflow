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
#ifndef ONEFLOW_CORE_VM_EP_RECORD_EVENT_INSTRUCTION_POLICY_H_
#define ONEFLOW_CORE_VM_EP_RECORD_EVENT_INSTRUCTION_POLICY_H_

#include <memory>
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/vm/ep_optional_event_record_status_querier.h"
#include "oneflow/core/vm/instruction_policy.h"
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/vm/ep_stream_policy_base.h"
#include "oneflow/core/vm/stream.h"

namespace oneflow {
namespace vm {
class EpRecordEventInstructionPolicy final : public InstructionPolicy {
 public:
  EpRecordEventInstructionPolicy(
      small_vector<intrusive::shared_ptr<LocalDepObject>>&& compute_local_dep_objects,
      const std::string& modifier)
      : compute_local_dep_objects_(std::move(compute_local_dep_objects)),
        modifier_(modifier),
        input_dependences_(),
        output_dependences_() {
    ForEachConstDependence([&](auto* dep) { input_dependences_.emplace_back(dep); });
    ForEachMutDependence([&](auto* dep) { output_dependences_.emplace_back(dep); });
    ForEachMut2Dependence([&](auto* dep) { output_dependences_.emplace_back(dep); });
  }

  ~EpRecordEventInstructionPolicy() override = default;
  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

  template<typename DoEachT>
  void ForEachConstDependence(const DoEachT& DoEach) const {
    if (modifier_ == "const") {
      for (const auto& dep : compute_local_dep_objects_) { DoEach(dep.get()); }
    }
  }

  template<typename DoEachT>
  void ForEachMutDependence(const DoEachT& DoEach) const {
    if (modifier_ == "mut") {
      for (const auto& dep : compute_local_dep_objects_) { DoEach(dep.get()); }
    }
  }

  template<typename DoEachT>
  void ForEachMut2Dependence(const DoEachT& DoEach) const {
    if (modifier_ == "mut2") {
      for (const auto& dep : compute_local_dep_objects_) { DoEach(dep.get()); }
    }
  }
  InstructionFuseType fuse_type() const override { return kEnableInstructionFuseAsTailOnly; }

  void InitInstructionStatus(Instruction* instruction) override {
    auto* status_buffer = instruction->mut_status_buffer();
    auto* stream = instruction->mut_stream();
    instruction->stream_policy().InitInstructionStatus(*stream, status_buffer);
    EpStreamPolicyBase* ep_stream_policy_base =
        dynamic_cast<EpStreamPolicyBase*>(stream->mut_stream_policy());
    CHECK_NOTNULL(ep_stream_policy_base);
    auto* ep_event_provider = ep_stream_policy_base->ep_event_provider();
    const auto& ep_event = CHECK_NOTNULL(ep_event_provider)->GetReusedEpEvent();
    auto* data_ptr = status_buffer->mut_buffer();
    EpOptionalEventRecordStatusQuerier::MutCast(data_ptr)->reset_ep_event(ep_event);
  }
  Maybe<void> Prepare(vm::Instruction* instruction) override { return Maybe<void>::Ok(); }
  std::string DebugName(const vm::Instruction&) const override { return "RecordEvent"; }
  void Compute(vm::Instruction* instruction) override {}

 private:
  small_vector<intrusive::shared_ptr<LocalDepObject>> compute_local_dep_objects_;
  const std::string modifier_;
  DependenceVector input_dependences_;
  DependenceVector output_dependences_;
};

}  // namespace vm

struct GetRecordEventInstructionPolicy : public StreamTypeVisitor<GetRecordEventInstructionPolicy> {
  template<typename... Args>
  static Maybe<vm::InstructionPolicy> VisitCompute(DeviceType device_type, Args&&... args) {
    return std::shared_ptr<vm::InstructionPolicy>(
        new vm::EpRecordEventInstructionPolicy(std::forward<Args>(args)...));
  }
  template<typename... Args>
  static Maybe<vm::InstructionPolicy> VisitHost2Device(DeviceType device_type, Args&&... args) {
    return std::shared_ptr<vm::InstructionPolicy>(
        new vm::EpRecordEventInstructionPolicy(std::forward<Args>(args)...));
  }
  template<typename... Args>
  static Maybe<vm::InstructionPolicy> VisitDevice2Host(DeviceType device_type, Args&&... args) {
    return std::shared_ptr<vm::InstructionPolicy>(
        new vm::EpRecordEventInstructionPolicy(std::forward<Args>(args)...));
  }
  template<typename... Args>
  static Maybe<vm::InstructionPolicy> VisitCcl(DeviceType device_type, Args&&... args) {
    return std::shared_ptr<vm::InstructionPolicy>(
        new vm::EpRecordEventInstructionPolicy(std::forward<Args>(args)...));
  }
  template<typename... Args>
  static Maybe<vm::InstructionPolicy> VisitBarrier(DeviceType device_type, Args&&... args) {
    UNIMPLEMENTED_THEN_RETURN() << "EpRecordEvent instruction not supported in Barrier stream";
  }
  template<typename... Args>
  static Maybe<vm::InstructionPolicy> VisitCriticalSection(DeviceType device_type, Args&&... args) {
    UNIMPLEMENTED_THEN_RETURN()
        << "EpRecordEvent instruction not supported in CriticalSection stream";
  }
  template<typename... Args>
  static Maybe<vm::InstructionPolicy> VisitLazyJobLauncher(DeviceType device_type, Args&&... args) {
    UNIMPLEMENTED_THEN_RETURN()
        << "EpRecordEvent instruction not supported in LaunchLazyJob stream";
  }
  template<typename... Args>
  static Maybe<vm::InstructionPolicy> VisitPinnedCompute(DeviceType device_type, Args&&... args) {
    return std::shared_ptr<vm::InstructionPolicy>(
        new vm::EpRecordEventInstructionPolicy(std::forward<Args>(args)...));
  }
};

}  // namespace oneflow
#endif  // ONEFLOW_CORE_EAGER_BLOB_INSTRUCTION_TYPE_H_
