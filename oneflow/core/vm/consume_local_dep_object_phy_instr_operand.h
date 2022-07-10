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
      small_vector<intrusive::shared_ptr<LocalDepObject>, kOpArgsReservedSize>&&
          compute_local_dep_objects,
      const std::string& modifier);
  ~ConsumeLocalDepObjectPhyInstrOperand() = default;

  const DependenceVector& input_dependences() const override { return input_dependences_; }
  const DependenceVector& output_dependences() const override { return output_dependences_; }

  template<typename DoEachT>
  void ForEachConstDependence(const DoEachT& DoEach) const;

  template<typename DoEachT>
  void ForEachMutDependence(const DoEachT& DoEach) const;

  template<typename DoEachT>
  void ForEachMut2Dependence(const DoEachT& DoEach) const;

  void ForEachInputEagerBlobObjects(void (*DoEach)(EagerBlobObject*)) const override {}

 private:
  small_vector<intrusive::shared_ptr<LocalDepObject>, kOpArgsReservedSize>
      compute_local_dep_objects_;
  const std::string modifier_;
  DependenceVector input_dependences_;
  DependenceVector output_dependences_;
};

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_CONSUME_LOCAL_DEP_OBJECT_H
