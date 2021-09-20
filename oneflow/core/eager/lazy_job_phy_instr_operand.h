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
#ifndef ONEFLOW_CORE_EAGER_LAZY_JOB_PHY_INSTR_OPERAND_H_
#define ONEFLOW_CORE_EAGER_LAZY_JOB_PHY_INSTR_OPERAND_H_

#include "oneflow/core/vm/instruction_operand.msg.h"
#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/eager/critical_section_phy_instr_operand.h"
#include "oneflow/core/framework/nn_graph_if.h"
#include "oneflow/core/common/notifier.h"

namespace oneflow {

namespace one {

using EagerBlobObjectListPtr =
    std::shared_ptr<const std::vector<std::shared_ptr<vm::EagerBlobObject>>>;

}

namespace vm {

class LaunchLazyJobPhyInstrOperand final : public PhyInstrOperand {
 public:
  LaunchLazyJobPhyInstrOperand(const LaunchLazyJobPhyInstrOperand&) = delete;
  LaunchLazyJobPhyInstrOperand(LaunchLazyJobPhyInstrOperand&&) = delete;
  ~LaunchLazyJobPhyInstrOperand() override = default;

  LaunchLazyJobPhyInstrOperand(
      const std::shared_ptr<InputCriticalSectionPhyInstrOperand>& inputs_critical_section,
      const std::shared_ptr<OutputCriticalSectionPhyInstrOperand>& outputs_critical_section,
      const std::shared_ptr<ParameterCriticalSectionPhyInstrOperand>& params_critical_section,
      const std::shared_ptr<NcclCriticalSectionPhyInstrOperand>& nccl_critical_section,
      const std::shared_ptr<NNGraphIf>& nn_graph)
      : inputs_critical_section_(inputs_critical_section),
        outputs_critical_section_(outputs_critical_section),
        params_critical_section_(params_critical_section),
        nccl_critical_section_(nccl_critical_section),
        nn_graph_(nn_graph) {}

  const std::shared_ptr<InputCriticalSectionPhyInstrOperand>& inputs_critical_section() const {
    return inputs_critical_section_;
  }
  const std::shared_ptr<OutputCriticalSectionPhyInstrOperand>& outputs_critical_section() const {
    return outputs_critical_section_;
  }
  const std::shared_ptr<ParameterCriticalSectionPhyInstrOperand>& params_critical_section() const {
    return params_critical_section_;
  }
  const std::shared_ptr<NcclCriticalSectionPhyInstrOperand>& nccl_critical_section() const {
    return nccl_critical_section_;
  }
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
  std::shared_ptr<ParameterCriticalSectionPhyInstrOperand> params_critical_section_;
  std::shared_ptr<NcclCriticalSectionPhyInstrOperand> nccl_critical_section_;
  std::shared_ptr<NNGraphIf> nn_graph_;
};
}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_LAZY_JOB_PHY_INSTR_OPERAND_H_
