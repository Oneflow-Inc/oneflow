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
#ifndef ONEFLOW_CORE_EAGER_CRITICAL_SECTION_PHY_INSTR_OPERAND_H_
#define ONEFLOW_CORE_EAGER_CRITICAL_SECTION_PHY_INSTR_OPERAND_H_

#include "oneflow/core/vm/phy_instr_operand.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/device/event_record.h"
#include "oneflow/core/framework/nn_graph_if.h"
#include "oneflow/core/common/buffer_manager.h"

namespace oneflow {

namespace one {

using EagerBlobObjectListPtr =
    std::shared_ptr<const std::vector<std::shared_ptr<vm::EagerBlobObject>>>;

}

namespace vm {

class Stream;

class CriticalSectionBeginPhyInstrOperand : public PhyInstrOperand {
 public:
  CriticalSectionBeginPhyInstrOperand(const CriticalSectionBeginPhyInstrOperand&) = delete;
  CriticalSectionBeginPhyInstrOperand(CriticalSectionBeginPhyInstrOperand&&) = delete;
  CriticalSectionBeginPhyInstrOperand& operator=(const CriticalSectionBeginPhyInstrOperand&) =
      delete;
  CriticalSectionBeginPhyInstrOperand& operator=(CriticalSectionBeginPhyInstrOperand&&) = delete;
  virtual ~CriticalSectionBeginPhyInstrOperand() = default;

  explicit CriticalSectionBeginPhyInstrOperand(
      const std::shared_ptr<NNGraphIf>& nn_graph,
      const one::EagerBlobObjectListPtr& eager_blob_objects,
      const std::shared_ptr<HashMap<std::string, std::shared_ptr<SharedEventRecord>>>&
          op_name2end_event_record,
      vm::Stream* vm_stream)
      : nn_graph_(nn_graph),
        eager_blob_objects_(eager_blob_objects),
        op_name2end_event_record_(op_name2end_event_record),
        vm_stream_(vm_stream) {}

  const std::shared_ptr<NNGraphIf>& nn_graph() const { return nn_graph_; }
  const one::EagerBlobObjectListPtr& eager_blob_objects() const { return eager_blob_objects_; }

  void ForEachMirroredObject(const std::function<void(vm::MirroredObject* compute)>&) const;

  void ForEachMutMirroredObject(const std::function<void(vm::MirroredObject* compute)>&) const;

  virtual const std::vector<std::string>& interfaces_op_names() const = 0;
  virtual const std::vector<bool>& interfaces_valid() const = 0;
  virtual std::string GetInterfaceBufferName(const std::string& job_name,
                                             const std::string& op_name) const = 0;
  virtual std::string GetInterfaceCriticalSectionCallbackBufferName(
      const std::string& job_name) const = 0;
  virtual std::string GetInterfaceCriticalSectionWaitBufferName(
      const std::string& job_name) const = 0;
  virtual void AccessBlobByOpName(uint64_t of_blob_ptr, const std::string& op_name) = 0;

  void FinishInvalidInterfaceEventRecords();
  void Finish();

  void ForEachInputEagerBlobObjects(void (*DoEach)(EagerBlobObject*)) const override {
    for (const auto& eager_blob_object : *eager_blob_objects_) { DoEach(eager_blob_object.get()); }
  }

 protected:
  std::shared_ptr<NNGraphIf> nn_graph_;
  one::EagerBlobObjectListPtr eager_blob_objects_;
  std::shared_ptr<HashMap<std::string, std::shared_ptr<SharedEventRecord>>>
      op_name2end_event_record_;
  HashMap<std::string, size_t> op_name2interface_index_;
  vm::Stream* vm_stream_;
};

class InputCriticalSectionBeginPhyInstrOperand final : public CriticalSectionBeginPhyInstrOperand {
 public:
  InputCriticalSectionBeginPhyInstrOperand(
      const std::shared_ptr<NNGraphIf>& nn_graph,
      const one::EagerBlobObjectListPtr& eager_blob_objects,
      const std::shared_ptr<HashMap<std::string, std::shared_ptr<SharedEventRecord>>>&
          op_name2end_event_record,
      vm::Stream* vm_stream)
      : CriticalSectionBeginPhyInstrOperand(nn_graph, eager_blob_objects, op_name2end_event_record,
                                            vm_stream),
        input_dependences_(),
        output_dependences_() {
    ForEachConstMirroredObject(SetInserter(&input_dependences_));
    ForEachMutMirroredObject(SetInserter(&output_dependences_));
    ForEachMut2MirroredObject(SetInserter(&output_dependences_));
    CHECK_EQ(nn_graph->inputs_op_names().size(), eager_blob_objects->size());
    CHECK_EQ(nn_graph->inputs_op_names().size(), nn_graph->inputs_valid().size());
    for (int i = 0; i < nn_graph->inputs_op_names().size(); ++i) {
      CHECK(op_name2interface_index_.emplace(nn_graph->inputs_op_names().at(i), i).second);
    }
  }

  ~InputCriticalSectionBeginPhyInstrOperand() override = default;

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

  // for inputs
  void ForEachConstMirroredObject(
      const std::function<void(vm::MirroredObject* compute)>& DoEach) const {
    ForEachMirroredObject(DoEach);
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
  void AccessBlobByOpName(uint64_t of_blob_ptr, const std::string& op_name) override;
  void ForEachMut2MirroredObject(const std::function<void(vm::MirroredObject* compute)>&) const {}

 private:
  DependenceVector input_dependences_;
  DependenceVector output_dependences_;
};

class OutputCriticalSectionBeginPhyInstrOperand final : public CriticalSectionBeginPhyInstrOperand {
 public:
  OutputCriticalSectionBeginPhyInstrOperand(
      const std::shared_ptr<NNGraphIf>& nn_graph,
      const one::EagerBlobObjectListPtr& eager_blob_objects,
      const std::shared_ptr<HashMap<std::string, std::shared_ptr<SharedEventRecord>>>&
          op_name2end_event_record,
      vm::Stream* vm_stream)
      : CriticalSectionBeginPhyInstrOperand(nn_graph, eager_blob_objects, op_name2end_event_record,
                                            vm_stream),
        input_dependences_(),
        output_dependences_() {
    ForEachConstMirroredObject(SetInserter(&input_dependences_));
    ForEachMutMirroredObject(SetInserter(&output_dependences_));
    ForEachMut2MirroredObject(SetInserter(&output_dependences_));
    CHECK_EQ(nn_graph->outputs_op_names().size(), eager_blob_objects->size());
    CHECK_EQ(nn_graph->outputs_op_names().size(), nn_graph->outputs_valid().size());
    for (int i = 0; i < nn_graph->outputs_op_names().size(); ++i) {
      CHECK(op_name2interface_index_.emplace(nn_graph->outputs_op_names().at(i), i).second);
    }
  }

  ~OutputCriticalSectionBeginPhyInstrOperand() override = default;

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

  // for inputs
  void ForEachConstMirroredObject(const std::function<void(vm::MirroredObject* compute)>&) const {}

  // for outputs
  void ForEachMut2MirroredObject(
      const std::function<void(vm::MirroredObject* compute)>& DoEach) const {
    ForEachMirroredObject(DoEach);
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
  void AccessBlobByOpName(uint64_t of_blob_ptr, const std::string& op_name) override;

 private:
  DependenceVector input_dependences_;
  DependenceVector output_dependences_;
};

class CriticalSectionEndPhyInstrOperand : public PhyInstrOperand {
 public:
  CriticalSectionEndPhyInstrOperand(const std::shared_ptr<EagerBlobObject>& eager_blob_object,
                                    const std::shared_ptr<SharedEventRecord>& event_record,
                                    vm::Stream* vm_stream)
      : eager_blob_object_(eager_blob_object), event_record_(event_record), vm_stream_(vm_stream) {}
  virtual ~CriticalSectionEndPhyInstrOperand() = default;

  const std::shared_ptr<SharedEventRecord>& event_record() const { return event_record_; }

  void ForEachMirroredObject(const std::function<void(vm::MirroredObject* compute)>&) const;

  void ForEachMutMirroredObject(const std::function<void(vm::MirroredObject* compute)>&) const;

  void ForEachInputEagerBlobObjects(void (*DoEach)(EagerBlobObject*)) const override {
    DoEach(eager_blob_object_.get());
  }

 private:
  std::shared_ptr<EagerBlobObject> eager_blob_object_;
  std::shared_ptr<SharedEventRecord> event_record_;
  vm::Stream* vm_stream_;
};

class InputCriticalSecondEndPhyInstrOperand final : public CriticalSectionEndPhyInstrOperand {
 public:
  InputCriticalSecondEndPhyInstrOperand(const std::shared_ptr<EagerBlobObject>& eager_blob_object,
                                        const std::shared_ptr<SharedEventRecord>& event_record,
                                        vm::Stream* vm_stream)
      : CriticalSectionEndPhyInstrOperand(eager_blob_object, event_record, vm_stream),
        input_dependences_(),
        output_dependences_() {
    ForEachConstMirroredObject(SetInserter(&input_dependences_));
    ForEachMutMirroredObject(SetInserter(&output_dependences_));
    ForEachMut2MirroredObject(SetInserter(&output_dependences_));
  }
  ~InputCriticalSecondEndPhyInstrOperand() override = default;

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

  void ForEachConstMirroredObject(
      const std::function<void(vm::MirroredObject* compute)>& DoEach) const {
    ForEachMirroredObject(DoEach);
  }

  void ForEachMut2MirroredObject(const std::function<void(vm::MirroredObject* compute)>&) const {}

 private:
  DependenceVector input_dependences_;
  DependenceVector output_dependences_;
};

class OutputCriticalSecondEndPhyInstrOperand final : public CriticalSectionEndPhyInstrOperand {
 public:
  OutputCriticalSecondEndPhyInstrOperand(const std::shared_ptr<EagerBlobObject>& eager_blob_object,
                                         const std::shared_ptr<SharedEventRecord>& event_record,
                                         vm::Stream* vm_stream)
      : CriticalSectionEndPhyInstrOperand(eager_blob_object, event_record, vm_stream),
        input_dependences_(),
        output_dependences_() {
    ForEachConstMirroredObject(SetInserter(&input_dependences_));
    ForEachMutMirroredObject(SetInserter(&output_dependences_));
    ForEachMut2MirroredObject(SetInserter(&output_dependences_));
  }
  ~OutputCriticalSecondEndPhyInstrOperand() override = default;

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

  // for inputs
  void ForEachConstMirroredObject(const std::function<void(vm::MirroredObject* compute)>&) const {}

  // for outputs
  void ForEachMut2MirroredObject(
      const std::function<void(vm::MirroredObject* compute)>& DoEach) const {
    ForEachMirroredObject(DoEach);
  }

 private:
  DependenceVector input_dependences_;
  DependenceVector output_dependences_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_CRITICAL_SECTION_PHY_INSTR_OPERAND_H_
