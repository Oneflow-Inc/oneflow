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
#ifndef ONEFLOW_CORE_VM_CRITICAL_SECTION_INSTRUCTION_POLICY_H_
#define ONEFLOW_CORE_VM_CRITICAL_SECTION_INSTRUCTION_POLICY_H_

#include "oneflow/core/common/buffer_manager.h"
#include "oneflow/core/device/event_record.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/nn_graph_if.h"
#include "oneflow/core/job/critical_section_instance.h"
#include "oneflow/core/vm/critical_section_status_querier.h"
#include "oneflow/core/vm/instruction.h"
#include "oneflow/core/vm/instruction_policy.h"
#include "oneflow/core/vm/instruction_policy_util.h"
#include "oneflow/core/vm/stream.h"

namespace oneflow {

namespace vm {

class CriticalSectionBeginInstructionPolicy
    : public InstructionPolicy,
      public std::enable_shared_from_this<CriticalSectionBeginInstructionPolicy> {
 public:
  CriticalSectionBeginInstructionPolicy(const CriticalSectionBeginInstructionPolicy&) = delete;
  CriticalSectionBeginInstructionPolicy(CriticalSectionBeginInstructionPolicy&&) = delete;
  CriticalSectionBeginInstructionPolicy& operator=(const CriticalSectionBeginInstructionPolicy&) =
      delete;
  CriticalSectionBeginInstructionPolicy& operator=(CriticalSectionBeginInstructionPolicy&&) =
      delete;
  virtual ~CriticalSectionBeginInstructionPolicy() = default;
  explicit CriticalSectionBeginInstructionPolicy(
      const std::shared_ptr<NNGraphIf>& nn_graph, const EagerBlobObjectListPtr& eager_blob_objects,
      const std::shared_ptr<HashMap<std::string, std::shared_ptr<SharedEventRecord>>>&
          op_name2end_event_record,
      Stream* vm_stream)
      : nn_graph_(nn_graph),
        eager_blob_objects_(eager_blob_objects),
        op_name2end_event_record_(op_name2end_event_record),
        vm_stream_(vm_stream) {}

  std::string DebugName(const Instruction& instruction) const override {
    return "CriticalSectionBegin";
  }
  Maybe<void> Prepare(Instruction* instruction) override { return Maybe<void>::Ok(); }
  void Compute(vm::Instruction* instruction) override {
    OF_PROFILER_RANGE_GUARD("CriticalSectionBegin");
    {
      const auto& critical_section_instance = MakeCriticalSectionInstance();
      const auto& job_name = critical_section_instance->job_name();
      auto* buffer_mgr = Singleton<BufferMgr<std::shared_ptr<CriticalSectionInstance>>>::Get();
      for (int i = 0; i < interfaces_op_names().size(); ++i) {
        if (interfaces_valid().at(i)) {
          const std::string& interface_op_name = interfaces_op_names().at(i);
          const auto& buffer_name = GetInterfaceBufferName(job_name, interface_op_name);
          buffer_mgr->Get(buffer_name)->Push(critical_section_instance);
        }
      }
      const auto& callback_buffer_name = GetInterfaceCriticalSectionCallbackBufferName(job_name);
      buffer_mgr->Get(callback_buffer_name)->Push(critical_section_instance);
      const auto& wait_buffer_name = GetInterfaceCriticalSectionWaitBufferName(job_name);
      buffer_mgr->Get(wait_buffer_name)->Push(critical_section_instance);
    }
    {
      auto* status_buffer_data = instruction->mut_status_buffer()->mut_buffer();
      auto* status_querier = CriticalSectionStatusQuerier::MutCast(status_buffer_data);
      status_querier->SetLaunched(std::make_shared<NaiveEventRecord>());
    }
  }
  const std::shared_ptr<NNGraphIf>& nn_graph() const { return nn_graph_; }
  const EagerBlobObjectListPtr& eager_blob_objects() const { return eager_blob_objects_; }

  void ForEachDependence(const std::function<void(Dependence* compute)>&) const;

  void ForEachMutDependence(const std::function<void(Dependence* compute)>&) const;

  virtual const std::vector<std::string>& interfaces_op_names() const = 0;
  virtual const std::vector<bool>& interfaces_valid() const = 0;
  virtual std::string GetInterfaceBufferName(const std::string& job_name,
                                             const std::string& op_name) const = 0;
  virtual std::string GetInterfaceCriticalSectionCallbackBufferName(
      const std::string& job_name) const = 0;
  virtual std::string GetInterfaceCriticalSectionWaitBufferName(
      const std::string& job_name) const = 0;
  virtual void AccessBlobByOpName(ep::Stream* stream, Blob* blob, const std::string& op_name) = 0;

  void FinishInvalidInterfaceEventRecords();
  void Finish();

 protected:
  std::shared_ptr<NNGraphIf> nn_graph_;
  EagerBlobObjectListPtr eager_blob_objects_;
  std::shared_ptr<HashMap<std::string, std::shared_ptr<SharedEventRecord>>>
      op_name2end_event_record_;
  HashMap<std::string, size_t> op_name2interface_index_;
  Stream* vm_stream_;

 private:
  class NaiveCriticalSectionInstance final : public CriticalSectionInstance {
   public:
    NaiveCriticalSectionInstance(const std::shared_ptr<CriticalSectionBeginInstructionPolicy>&
                                     critical_section_begin_instruction_policy,
                                 const std::string& job_name)
        : CriticalSectionInstance(),
          critical_section_begin_instruction_policy_(critical_section_begin_instruction_policy),
          job_name_(job_name) {}

    ~NaiveCriticalSectionInstance() override = default;

    const std::string& job_name() const override { return job_name_; }

    void AccessBlobByOpName(ep::Stream* stream, Blob* blob,
                            const std::string& op_name) const override {
      critical_section_begin_instruction_policy_->AccessBlobByOpName(stream, blob, op_name);
    }
    void Finish() const override { critical_section_begin_instruction_policy_->Finish(); }

   private:
    std::shared_ptr<CriticalSectionBeginInstructionPolicy>
        critical_section_begin_instruction_policy_;
    std::string job_name_;
  };

  std::shared_ptr<CriticalSectionInstance> MakeCriticalSectionInstance() {
    return std::make_shared<NaiveCriticalSectionInstance>(this->shared_from_this(),
                                                          nn_graph_->job_name());
  }
};

class InputCriticalSectionBeginInstructionPolicy final
    : public CriticalSectionBeginInstructionPolicy {
 public:
  InputCriticalSectionBeginInstructionPolicy(
      const std::shared_ptr<NNGraphIf>& nn_graph, const EagerBlobObjectListPtr& eager_blob_objects,
      const std::shared_ptr<HashMap<std::string, std::shared_ptr<SharedEventRecord>>>&
          op_name2end_event_record,
      Stream* vm_stream)
      : CriticalSectionBeginInstructionPolicy(nn_graph, eager_blob_objects,
                                              op_name2end_event_record, vm_stream),
        input_dependences_(),
        output_dependences_() {
    ForEachConstDependence(InstructionPolicyUtil::SetInserter(&input_dependences_));
    ForEachMutDependence(InstructionPolicyUtil::SetInserter(&output_dependences_));
    ForEachMut2Dependence(InstructionPolicyUtil::SetInserter(&output_dependences_));
    CHECK_EQ(nn_graph->inputs_op_names().size(), eager_blob_objects->size());
    CHECK_EQ(nn_graph->inputs_op_names().size(), nn_graph->inputs_valid().size());
    for (int i = 0; i < nn_graph->inputs_op_names().size(); ++i) {
      CHECK(op_name2interface_index_.emplace(nn_graph->inputs_op_names().at(i), i).second);
    }
  }

  ~InputCriticalSectionBeginInstructionPolicy() override = default;

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

  // for inputs
  void ForEachConstDependence(const std::function<void(Dependence* compute)>& DoEach) const {
    ForEachDependence(DoEach);
  }

  // for outputs
  const std::vector<std::string>& interfaces_op_names() const override {
    return nn_graph_->inputs_op_names();
  }
  const std::vector<bool>& interfaces_valid() const override { return nn_graph_->inputs_valid(); }
  std::string GetInterfaceBufferName(const std::string& job_name,
                                     const std::string& op_name) const override {
    return GetInputBufferName(job_name, op_name);
  }
  std::string GetInterfaceCriticalSectionCallbackBufferName(
      const std::string& job_name) const override {
    return GetInputCriticalSectionCallbackBufferName(job_name);
  }
  std::string GetInterfaceCriticalSectionWaitBufferName(
      const std::string& job_name) const override {
    return GetInputCriticalSectionWaitBufferName(job_name);
  }
  void AccessBlobByOpName(ep::Stream* stream, Blob* blob, const std::string& op_name) override;
  void ForEachMut2Dependence(const std::function<void(Dependence* compute)>&) const {}

 private:
  DependenceVector input_dependences_;
  DependenceVector output_dependences_;
};

class OutputCriticalSectionBeginInstructionPolicy final
    : public CriticalSectionBeginInstructionPolicy {
 public:
  OutputCriticalSectionBeginInstructionPolicy(
      const std::shared_ptr<NNGraphIf>& nn_graph, const EagerBlobObjectListPtr& eager_blob_objects,
      const std::shared_ptr<HashMap<std::string, std::shared_ptr<SharedEventRecord>>>&
          op_name2end_event_record,
      Stream* vm_stream)
      : CriticalSectionBeginInstructionPolicy(nn_graph, eager_blob_objects,
                                              op_name2end_event_record, vm_stream),
        input_dependences_(),
        output_dependences_() {
    ForEachConstDependence(InstructionPolicyUtil::SetInserter(&input_dependences_));
    ForEachMutDependence(InstructionPolicyUtil::SetInserter(&output_dependences_));
    ForEachMut2Dependence(InstructionPolicyUtil::SetInserter(&output_dependences_));
    CHECK_EQ(nn_graph->outputs_op_names().size(), eager_blob_objects->size());
    CHECK_EQ(nn_graph->outputs_op_names().size(), nn_graph->outputs_valid().size());
    for (int i = 0; i < nn_graph->outputs_op_names().size(); ++i) {
      CHECK(op_name2interface_index_.emplace(nn_graph->outputs_op_names().at(i), i).second);
    }
  }

  ~OutputCriticalSectionBeginInstructionPolicy() override = default;

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

  // for inputs
  void ForEachConstDependence(const std::function<void(Dependence* compute)>&) const {}

  // for outputs
  void ForEachMut2Dependence(const std::function<void(Dependence* compute)>& DoEach) const {
    ForEachDependence(DoEach);
  }

  const std::vector<std::string>& interfaces_op_names() const override {
    return nn_graph_->outputs_op_names();
  }
  const std::vector<bool>& interfaces_valid() const override { return nn_graph_->outputs_valid(); }
  std::string GetInterfaceBufferName(const std::string& job_name,
                                     const std::string& op_name) const override {
    return GetOutputBufferName(job_name, op_name);
  }
  std::string GetInterfaceCriticalSectionCallbackBufferName(
      const std::string& job_name) const override {
    return GetOutputCriticalSectionCallbackBufferName(job_name);
  }
  std::string GetInterfaceCriticalSectionWaitBufferName(
      const std::string& job_name) const override {
    return GetOutputCriticalSectionWaitBufferName(job_name);
  }
  void AccessBlobByOpName(ep::Stream* stream, Blob* blob, const std::string& op_name) override;

 private:
  DependenceVector input_dependences_;
  DependenceVector output_dependences_;
};

class CriticalSectionEndInstructionPolicy : public InstructionPolicy {
 public:
  CriticalSectionEndInstructionPolicy(const CriticalSectionEndInstructionPolicy&) = delete;
  CriticalSectionEndInstructionPolicy(CriticalSectionEndInstructionPolicy&&) = delete;
  CriticalSectionEndInstructionPolicy& operator=(const CriticalSectionEndInstructionPolicy&) =
      delete;
  CriticalSectionEndInstructionPolicy& operator=(CriticalSectionEndInstructionPolicy&&) = delete;
  CriticalSectionEndInstructionPolicy(const std::shared_ptr<EagerBlobObject>& eager_blob_object,
                                      const std::shared_ptr<SharedEventRecord>& event_record,
                                      vm::Stream* vm_stream)
      : eager_blob_object_(eager_blob_object), event_record_(event_record), vm_stream_(vm_stream) {}
  virtual ~CriticalSectionEndInstructionPolicy() = default;

  std::string DebugName(const Instruction& instruction) const override {
    return "CriticalSectionEnd";
  }
  Maybe<void> Prepare(Instruction* instruction) override { return Maybe<void>::Ok(); }
  void Compute(Instruction* instruction) override {
    auto* status_buffer_data = instruction->mut_status_buffer()->mut_buffer();
    auto* status_querier = CriticalSectionStatusQuerier::MutCast(status_buffer_data);
    status_querier->SetLaunched(event_record());
  }
  const std::shared_ptr<SharedEventRecord>& event_record() const { return event_record_; }

  void ForEachDependence(const std::function<void(vm::Dependence* compute)>&) const;

  void ForEachMutDependence(const std::function<void(vm::Dependence* compute)>&) const;

 private:
  std::shared_ptr<EagerBlobObject> eager_blob_object_;
  std::shared_ptr<SharedEventRecord> event_record_;
  vm::Stream* vm_stream_;
};

class InputCriticalSectionEndInstructionPolicy final : public CriticalSectionEndInstructionPolicy {
 public:
  InputCriticalSectionEndInstructionPolicy(
      const std::shared_ptr<EagerBlobObject>& eager_blob_object,
      const std::shared_ptr<SharedEventRecord>& event_record, vm::Stream* vm_stream)
      : CriticalSectionEndInstructionPolicy(eager_blob_object, event_record, vm_stream),
        input_dependences_(),
        output_dependences_() {
    ForEachConstDependence(InstructionPolicyUtil::SetInserter(&input_dependences_));
    ForEachMutDependence(InstructionPolicyUtil::SetInserter(&output_dependences_));
    ForEachMut2Dependence(InstructionPolicyUtil::SetInserter(&output_dependences_));
  }
  ~InputCriticalSectionEndInstructionPolicy() override = default;

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

  void ForEachConstDependence(const std::function<void(vm::Dependence* compute)>& DoEach) const {
    ForEachDependence(DoEach);
  }

  void ForEachMut2Dependence(const std::function<void(vm::Dependence* compute)>&) const {}

 private:
  DependenceVector input_dependences_;
  DependenceVector output_dependences_;
};

class OutputCriticalSectionEndInstructionPolicy final : public CriticalSectionEndInstructionPolicy {
 public:
  OutputCriticalSectionEndInstructionPolicy(
      const std::shared_ptr<EagerBlobObject>& eager_blob_object,
      const std::shared_ptr<SharedEventRecord>& event_record, vm::Stream* vm_stream)
      : CriticalSectionEndInstructionPolicy(eager_blob_object, event_record, vm_stream),
        input_dependences_(),
        output_dependences_() {
    ForEachConstDependence(InstructionPolicyUtil::SetInserter(&input_dependences_));
    ForEachMutDependence(InstructionPolicyUtil::SetInserter(&output_dependences_));
    ForEachMut2Dependence(InstructionPolicyUtil::SetInserter(&output_dependences_));
  }
  ~OutputCriticalSectionEndInstructionPolicy() override = default;

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

  // for inputs
  void ForEachConstDependence(const std::function<void(vm::Dependence* compute)>&) const {}

  // for outputs
  void ForEachMut2Dependence(const std::function<void(vm::Dependence* compute)>& DoEach) const {
    ForEachDependence(DoEach);
  }

 private:
  DependenceVector input_dependences_;
  DependenceVector output_dependences_;
};

}  // namespace vm
}  // namespace oneflow
#endif  // ONEFLOW_CORE_VM_CRITICAL_SECTION_INSTRUCTION_POLICY_H_
