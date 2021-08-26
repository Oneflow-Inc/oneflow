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
#include "oneflow/core/eager/run_lazy_job_phy_instr_operand.h"

namespace oneflow {
namespace vm {

void RunLazyJobPhyInstrOperand::ForEachConstMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  for (const auto& input : *inputs()) {
    DoEach(nullptr, CHECK_JUST(input->compute_local_dep_object())->mut_mirrored_object());
  }
}

void RunLazyJobPhyInstrOperand::ForEachMutMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  for (const auto& parameter : *parameters()) {
    DoEach(nullptr, CHECK_JUST(parameter->compute_local_dep_object())->mut_mirrored_object());
  }
}

void RunLazyJobPhyInstrOperand::ForEachMut2MirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  // TODO(lixinqi): move partial of outputs into ForEachMutMirroredObject if shape infered before
  // compute.
  for (const auto& output : *outputs()) {
    DoEach(nullptr, CHECK_JUST(output->compute_local_dep_object())->mut_mirrored_object());
  }
}

}  // namespace vm
}  // namespace oneflow
