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
#include "oneflow/core/framework/stream_is_comm_net_stream.h"

namespace oneflow {
namespace vm {

LocalCallOpKernelPhyInstrOperand::LocalCallOpKernelPhyInstrOperand(
    const std::shared_ptr<one::StatefulLocalOpKernel>& opkernel,
    const one::EagerBlobObjectListPtr& inputs, const one::EagerBlobObjectListPtr& outputs,
    const std::shared_ptr<const one::ConsistentTensorInferResult>& consistent_tensor_infer_result,
    const one::OpExprInterpContext& op_interp_ctx,
    const one::DevVmDepObjectConsumeMode dev_vm_dep_object_consume_mode)
    : call_ctx_{
        .composed_attrs = ComposedAttrMap(op_interp_ctx.attrs, opkernel->base_attrs()),
        .inputs = inputs,
        .outputs = outputs,
        .consistent_tensor_infer_result = consistent_tensor_infer_result,
        .op_interp_ctx = op_interp_ctx,
      },
      opkernel_(opkernel),
      user_opkernel_(nullptr),
      need_temp_storage_(false),
      dev_vm_dep_object_consume_mode_(dev_vm_dep_object_consume_mode),
      input_dependences_(),
      output_dependences_() {
  ForEachConstMirroredObject(SetInserter(&input_dependences_));
  ForEachMutMirroredObject(SetInserter(&output_dependences_));
  ForEachMut2MirroredObject(SetInserter(&output_dependences_));
  InitStreamSequentialDependence();
}

Maybe<void> LocalCallOpKernelPhyInstrOperand::Init() {
  return mut_opkernel()->ChooseOpKernel(&call_ctx_, &user_opkernel_, &need_temp_storage_);
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
  const auto& stream = opkernel().stream();
  auto* device_schedule_dep_object = stream->mut_schedule_local_dep_object();
  if (StreamRoleSwitch<IsCommNetStream>(stream->stream_role())) {
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
  const auto& stream = opkernel().stream();
  const auto& opt_transport_dep_object = stream->mut_transport_local_dep_object();
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
