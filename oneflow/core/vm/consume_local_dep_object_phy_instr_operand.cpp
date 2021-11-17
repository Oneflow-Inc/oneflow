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

void ConsumeLocalDepObjectPhyInstrOperand::ForEachConstMirroredObject(
    const std::function<void(MirroredObject* compute)>& DoEach) const {
  if (modifier_ == "const") { DoEach(compute_local_dep_object_->mut_mirrored_object()); }
}

void ConsumeLocalDepObjectPhyInstrOperand::ForEachMutMirroredObject(
    const std::function<void(MirroredObject* compute)>& DoEach) const {
  if (modifier_ == "mut") { DoEach(compute_local_dep_object_->mut_mirrored_object()); }
}

void ConsumeLocalDepObjectPhyInstrOperand::ForEachMut2MirroredObject(
    const std::function<void(MirroredObject* compute)>& DoEach) const {
  if (modifier_ == "mut2") { DoEach(compute_local_dep_object_->mut_mirrored_object()); }
}

}  // namespace vm
}  // namespace oneflow
