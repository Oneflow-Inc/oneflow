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
#include "oneflow/core/eager/dev_vm_dep_object_consume_mode.h"

namespace oneflow {
namespace vm {

Maybe<void> LocalCallOpKernelPhyInstrOperand::Init() {
  JUST(mut_opkernel()->ChooseOpKernel(&user_opkernel_, &need_temp_storage_, attrs(), inputs().get(),
                                      outputs().get(), consistent_tensor_infer_result().get()));
  return Maybe<void>::Ok();
}

void LocalCallOpKernelPhyInstrOperand::ForEachConstMirroredObject(
    const std::function<void(vm::MirroredObject* compute)>& DoEach) const {
  const auto& input_list = inputs();
  for (int64_t index : opkernel().input_tuple_indexes4const_ibns()) {
    const auto& input = input_list->at(index);
    DoEach(CHECK_JUST(input->compute_local_dep_object()));
  }
}

void LocalCallOpKernelPhyInstrOperand::InitStreamSequentialDependence() {
  const auto& device = opkernel().device();
  auto* device_schedule_dep_object = device->mut_schedule_local_dep_object();
  if (device->type() == "async_launched_nccl") {
    // Sequantialize nccl instructions to avoid deadlock
    stream_sequential_dependence_ = device_schedule_dep_object;
  } else {
    // Sequantialize instructions to avoid explosive memory allocation of source ops
    if (dev_vm_dep_object_consume_mode() == one::DevVmDepObjectConsumeMode::MUTABLE) {
      stream_sequential_dependence_ = device_schedule_dep_object;
    } else if (opkernel().input_tuple_indexes4const_ibns().empty()
               && opkernel().input_tuple_indexes4mut_ibns().empty()) {
      stream_sequential_dependence_ = device_schedule_dep_object;
    }
  }
}

void LocalCallOpKernelPhyInstrOperand::ForEachMutMirroredObject(
    const std::function<void(vm::MirroredObject* compute)>& DoEach) const {
  const auto& device = opkernel().device();
  const auto& opt_transport_dep_object = device->mut_transport_local_dep_object();
  if (opt_transport_dep_object.has_value()) { DoEach(CHECK_JUST(opt_transport_dep_object)); }

  const auto& input_list = inputs();
  for (int64_t index : opkernel().input_tuple_indexes4mut_ibns()) {
    const auto& input = input_list->at(index);
    DoEach(CHECK_JUST(input->compute_local_dep_object()));
  }
  const auto& output_list = outputs();
  for (int64_t index : opkernel().output_tuple_indexes4mut_obns()) {
    const auto& output = output_list->at(index);
    DoEach(CHECK_JUST(output->compute_local_dep_object()));
  }
}

void LocalCallOpKernelPhyInstrOperand::ForEachMut2MirroredObject(
    const std::function<void(vm::MirroredObject* compute)>& DoEach) const {
  const auto& output_list = outputs();
  for (int64_t index : opkernel().output_tuple_indexes4mut2_obns()) {
    const auto& output = output_list->at(index);
    DoEach(CHECK_JUST(output->compute_local_dep_object()));
  }
}

}  // namespace vm
}  // namespace oneflow
