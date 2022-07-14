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
#ifndef ONEFLOW_CORE_VM_CONSUME_LOCAL_DEP_OBJECT_H
#define ONEFLOW_CORE_VM_CONSUME_LOCAL_DEP_OBJECT_H

#include <functional>
#include "oneflow/core/vm/phy_instr_operand.h"
#include "oneflow/core/eager/local_dep_object.h"

namespace oneflow {

namespace vm {

class ConsumeLocalDepObjectPhyInstrOperand : public PhyInstrOperand {
 public:
  ConsumeLocalDepObjectPhyInstrOperand(
      std::vector<intrusive::shared_ptr<LocalDepObject>>&& compute_local_dep_objects,
      const std::string& modifier)
      : compute_local_dep_objects_(std::move(compute_local_dep_objects)),
        modifier_(modifier),
        input_dependences_(),
        output_dependences_() {
    ForEachConstDependence(SetInserter(&input_dependences_));
    ForEachMutDependence(SetInserter(&output_dependences_));
    ForEachMut2Dependence(SetInserter(&output_dependences_));
    stream_sequential_dependence_ = nullptr;
  }

  ~ConsumeLocalDepObjectPhyInstrOperand() = default;

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

  void ForEachConstDependence(const std::function<void(Dependence* compute)>&) const;

  void ForEachMutDependence(const std::function<void(Dependence* compute)>&) const;

  void ForEachMut2Dependence(const std::function<void(Dependence* compute)>&) const;

  void ForEachInputEagerBlobObjects(void (*DoEach)(EagerBlobObject*)) const override {}

 private:
  std::vector<intrusive::shared_ptr<LocalDepObject>> compute_local_dep_objects_;
  const std::string modifier_;
  DependenceVector input_dependences_;
  DependenceVector output_dependences_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CONSUME_LOCAL_DEP_OBJECT_H
