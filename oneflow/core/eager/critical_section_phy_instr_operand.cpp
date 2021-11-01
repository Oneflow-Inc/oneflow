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
#include "oneflow/core/eager/critical_section_phy_instr_operand.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/common/decorator.h"

namespace oneflow {
namespace vm {

void CriticalSectionBeginPhyInstrOperand::ForEachMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  for (const auto& eager_blob_object : *eager_blob_objects_) {
    DoEach(nullptr,
           CHECK_JUST(eager_blob_object->compute_local_dep_object())->mut_mirrored_object());
  }
}

void CriticalSectionEndPhyInstrOperand::ForEachMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  DoEach(nullptr,
         CHECK_JUST(eager_blob_object_->compute_local_dep_object())->mut_mirrored_object());
}

namespace {

Maybe<LocalDepObject*> RawCriticalSectionLocalDepObject() {
  return JUST(Device::New("critical_section"))->mut_schedule_local_dep_object();
}

constexpr auto* CriticalSectionLocalDepObject =
    DECORATE(&RawCriticalSectionLocalDepObject, ThreadLocal);

}  // namespace

void CriticalSectionBeginPhyInstrOperand::ForEachMutMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  DoEach(nullptr, CHECK_JUST(CriticalSectionLocalDepObject())->mut_mirrored_object());
  DoEach(nullptr, local_dep_object_->mut_mirrored_object());
}

void CriticalSectionEndPhyInstrOperand::ForEachMutMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  DoEach(nullptr, CHECK_JUST(CriticalSectionLocalDepObject())->mut_mirrored_object());
}

}  // namespace vm
}  // namespace oneflow
