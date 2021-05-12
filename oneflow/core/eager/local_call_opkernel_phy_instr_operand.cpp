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
#include "oneflow/core/eager/local_call_opkernel_phy_instr_operand.h"
#include "oneflow/user/kernels/stateful_local_opkernel.h"

namespace oneflow {
namespace vm {

void LocalCallOpKernelPhyInstrOperand::ForEachConstMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  const auto& input_list = inputs();
  for (int64_t index : opkernel().input_tuple_indexes4const_ibns()) {
    const auto& input = input_list->at(index);
    DoEach(
        CHECK_JUST(input->infer_local_dep_object())->mut_local_dep_object()->mut_mirrored_object(),
        CHECK_JUST(input->compute_local_dep_object())
            ->mut_local_dep_object()
            ->mut_mirrored_object());
  }
}

void LocalCallOpKernelPhyInstrOperand::ForEachMutMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  // Sequantialize instructions in the same stream by consuming `compute_local_dep_object` of the same device.
  DoEach(opkernel().infer_local_dep_object()->mut_local_dep_object()->mut_mirrored_object(),
         opkernel().device()->mut_compute_local_dep_object()->mut_mirrored_object());

  const auto& input_list = inputs();
  for (int64_t index : opkernel().input_tuple_indexes4mut_ibns()) {
    const auto& input = input_list->at(index);
    DoEach(
        CHECK_JUST(input->infer_local_dep_object())->mut_local_dep_object()->mut_mirrored_object(),
        CHECK_JUST(input->compute_local_dep_object())
            ->mut_local_dep_object()
            ->mut_mirrored_object());
  }
  const auto& output_list = outputs();
  for (int64_t index : opkernel().output_tuple_indexes4mut_obns()) {
    const auto& output = output_list->at(index);
    DoEach(
        CHECK_JUST(output->infer_local_dep_object())->mut_local_dep_object()->mut_mirrored_object(),
        CHECK_JUST(output->compute_local_dep_object())
            ->mut_local_dep_object()
            ->mut_mirrored_object());
  }
}

void LocalCallOpKernelPhyInstrOperand::ForEachMut2MirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>& DoEach)
    const {
  const auto& output_list = outputs();
  for (int64_t index : opkernel().output_tuple_indexes4mut2_obns()) {
    const auto& output = output_list->at(index);
    DoEach(
        CHECK_JUST(output->infer_local_dep_object())->mut_local_dep_object()->mut_mirrored_object(),
        CHECK_JUST(output->compute_local_dep_object())
            ->mut_local_dep_object()
            ->mut_mirrored_object());
  }
}

}  // namespace vm
}  // namespace oneflow
