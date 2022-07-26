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
#include "oneflow/core/vm/consume_local_dep_object_phy_instr_operand.h"
#include "oneflow/core/eager/local_dep_object.h"

namespace oneflow {

namespace vm {

ConsumeLocalDepObjectPhyInstrOperand::ConsumeLocalDepObjectPhyInstrOperand(
    small_vector<intrusive::shared_ptr<LocalDepObject>, kOpArgsReservedSize>&&
        compute_local_dep_objects,
    const std::string& modifier)
    : compute_local_dep_objects_(std::move(compute_local_dep_objects)),
      modifier_(modifier),
      input_dependences_(),
      output_dependences_() {
  ForEachConstDependence([&](auto* dep) { input_dependences_.emplace_back(dep); });
  ForEachMutDependence([&](auto* dep) { output_dependences_.emplace_back(dep); });
  ForEachMut2Dependence([&](auto* dep) { output_dependences_.emplace_back(dep); });
  stream_sequential_dependence_ = nullptr;
}
template<typename DoEachT>
void ConsumeLocalDepObjectPhyInstrOperand::ForEachConstDependence(const DoEachT& DoEach) const {
  if (modifier_ == "const") {
    for (const auto& dep : compute_local_dep_objects_) { DoEach(dep.get()); }
  }
}

template<typename DoEachT>
void ConsumeLocalDepObjectPhyInstrOperand::ForEachMutDependence(const DoEachT& DoEach) const {
  if (modifier_ == "mut") {
    for (const auto& dep : compute_local_dep_objects_) { DoEach(dep.get()); }
  }
}

template<typename DoEachT>
void ConsumeLocalDepObjectPhyInstrOperand::ForEachMut2Dependence(const DoEachT& DoEach) const {
  if (modifier_ == "mut2") {
    for (const auto& dep : compute_local_dep_objects_) { DoEach(dep.get()); }
  }
}

}  // namespace vm
}  // namespace oneflow
