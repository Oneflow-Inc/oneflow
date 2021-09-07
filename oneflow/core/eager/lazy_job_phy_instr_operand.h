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
#ifndef ONEFLOW_CORE_EAGER_RUN_JOB_PHY_INSTR_OPERAND_H_
#define ONEFLOW_CORE_EAGER_RUN_JOB_PHY_INSTR_OPERAND_H_

#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/framework/nn_graph_if.h"
#include "oneflow/core/common/notifier.h"

namespace oneflow {

namespace one {

using EagerBlobObjectListPtr =
    std::shared_ptr<const std::vector<std::shared_ptr<vm::EagerBlobObject>>>;

}

namespace vm {

class CriticalSectionPhyInstrOperand : public PhyInstrOperand {
 public:
  CriticalSectionPhyInstrOperand(const CriticalSectionPhyInstrOperand&) = delete;
  CriticalSectionPhyInstrOperand(CriticalSectionPhyInstrOperand&&) = delete;
  virtual ~CriticalSectionPhyInstrOperand() = default;

  CriticalSectionPhyInstrOperand(const one::EagerBlobObjectListPtr& eager_blob_objects,
                            const std::shared_ptr<NNGraphIf>& nn_graph);

  const std::shared_ptr<NNGraphIf>& nn_graph() const { return nn_graph_; }

  // Called by producer.
  void ProducerNotifyConsumerAndWaitConsumerAck() const;

  // Called by consumers.
  void ConsumerFetchBlobAndNotifyProducerAck(const std::string& op_name, const std::function<void(Blob*)>& Callback) const;

  virtual const std::vector<std::string>& op_names() const = 0;

  void ForEachMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&) const;

  void ForEachMutMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override {}

 protected:
  one::EagerBlobObjectListPtr eager_blob_objects_;
  std::shared_ptr<NNGraphIf> nn_graph_;
  // op_name to index within `eager_blob_objects_`.
  HashMap<std::string, int64_t> op_name2index_; 
  // producers notify consumers by op_name2producer_notifier_.
  HashMap<std::string, std::unique_ptr<Notifier>> op_name2producer_notifier_;
  // consumers notify producers by op_name2producer_notifier_.
  HashMap<std::string, std::unique_ptr<Notifier>> op_name2consumer_notifier_;
};

class InputCriticalSectionPhyInstrOperand final : public CriticalSectionPhyInstrOperand {
 public:
  using CriticalSectionPhyInstrOperand::CriticalSectionPhyInstrOperand;
  ~InputCriticalSectionPhyInstrOperand() override = default;

  const std::vector<std::string>& op_names() const override { return nn_graph_->input_op_names(); }

  // for inputs
  void ForEachConstMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
      const override { ForEachMirroredObject(DoEach); }

  // for outputs
  void ForEachMut2MirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override {}

};

class OutputCriticalSectionPhyInstrOperand final : public CriticalSectionPhyInstrOperand {
 public:
  using CriticalSectionPhyInstrOperand::CriticalSectionPhyInstrOperand;
  ~OutputCriticalSectionPhyInstrOperand() override = default;

  const std::vector<std::string>& op_names() const override { return nn_graph_->output_op_names(); }

  // for inputs
  void ForEachConstMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
      const override {}

  // for outputs
  void ForEachMut2MirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override { ForEachMirroredObject(DoEach); }

};

class LaunchLazyJobPhyInstrOperand final : public PhyInstrOperand {
 public:
  LaunchLazyJobPhyInstrOperand(const LaunchLazyJobPhyInstrOperand&) = delete;
  LaunchLazyJobPhyInstrOperand(LaunchLazyJobPhyInstrOperand&&) = delete;
  ~LaunchLazyJobPhyInstrOperand() override = default;

  LaunchLazyJobPhyInstrOperand(
      const std::shared_ptr<InputCriticalSectionPhyInstrOperand>& inputs_critical_section,
      const std::shared_ptr<OutputCriticalSectionPhyInstrOperand>& outputs_critical_section,
                            const one::EagerBlobObjectListPtr& parameters,
                            const std::shared_ptr<NNGraphIf>& nn_graph)
      : inputs_critical_section_(inputs_critical_section), outputs_critical_section_(outputs_critical_section), parameters_(parameters), nn_graph_(nn_graph) {}

  const std::shared_ptr<InputCriticalSectionPhyInstrOperand>& inputs_critical_section() const { return inputs_critical_section_; }
  const std::shared_ptr<OutputCriticalSectionPhyInstrOperand>& outputs_critical_section() const { return outputs_critical_section_; }
  const one::EagerBlobObjectListPtr& parameters() const { return parameters_; }
  const std::shared_ptr<NNGraphIf>& nn_graph() const { return nn_graph_; }

  void ForEachConstMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override {
    // Do nothing because lifetime of inputs are managed by inputs_critical_section_.
  }

  void ForEachMutMirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override;

  void ForEachMut2MirroredObject(
      const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&)
      const override {
    // Do nothing because lifetime of outputs are managed by outputs_critical_section_.
  }

 private:
  std::shared_ptr<InputCriticalSectionPhyInstrOperand> inputs_critical_section_;
  std::shared_ptr<OutputCriticalSectionPhyInstrOperand> outputs_critical_section_;
  one::EagerBlobObjectListPtr parameters_;
  std::shared_ptr<NNGraphIf> nn_graph_;
};
}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_RUN_JOB_PHY_INSTR_OPERAND_H_
