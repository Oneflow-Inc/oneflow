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

#include "oneflow/core/eager/eager_blob_object.h"
#include "oneflow/core/eager/local_dep_object.h"
#include "oneflow/core/device/event_record.h"
#include "oneflow/core/eager/critical_section_phy_instr_operand.h"
#include "oneflow/core/framework/nn_graph_if.h"
#include "oneflow/core/common/notifier.h"

namespace oneflow {

namespace vm {

class LaunchLazyJobPhyInstrOperand final : public PhyInstrOperand {
 public:
  LaunchLazyJobPhyInstrOperand(const LaunchLazyJobPhyInstrOperand&) = delete;
  LaunchLazyJobPhyInstrOperand(LaunchLazyJobPhyInstrOperand&&) = delete;
  ~LaunchLazyJobPhyInstrOperand() override = default;

  LaunchLazyJobPhyInstrOperand(const std::shared_ptr<NNGraphIf>& nn_graph,
                               const vm::EagerBlobObjectListPtr& param_blob_objects)
      : nn_graph_(nn_graph),
        param_blob_objects_(param_blob_objects),
        input_dependences_(),
        output_dependences_() {
    ForEachConstDependence(SetInserter(&input_dependences_));
    ForEachMutDependence(SetInserter(&output_dependences_));
    ForEachMut2Dependence(SetInserter(&output_dependences_));
    stream_sequential_dependence_ = nullptr;
  }

  const std::shared_ptr<NNGraphIf>& nn_graph() const { return nn_graph_; }

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

  void ForEachConstDependence(const std::function<void(vm::Dependence* compute)>&) const {}

  void ForEachMutDependence(const std::function<void(vm::Dependence* compute)>&) const;

  void ForEachMut2Dependence(const std::function<void(vm::Dependence* compute)>&) const {}

  void ForEachInputEagerBlobObjects(void (*DoEach)(EagerBlobObject*)) const override {
    for (const auto& eager_blob_object : *param_blob_objects_) { DoEach(eager_blob_object.get()); }
  }

 private:
  std::shared_ptr<NNGraphIf> nn_graph_;
  vm::EagerBlobObjectListPtr param_blob_objects_;
  DependenceVector input_dependences_;
  DependenceVector output_dependences_;
};
}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_EAGER_LAZY_JOB_PHY_INSTR_OPERAND_H_
