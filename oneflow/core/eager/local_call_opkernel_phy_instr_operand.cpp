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
namespace eager {
void LocalCallOpKernelPhyInstrOperand::ForEachConstMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&) const {}

void LocalCallOpKernelPhyInstrOperand::ForEachMutMirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&) const {}

void LocalCallOpKernelPhyInstrOperand::ForEachMut2MirroredObject(
    const std::function<void(vm::MirroredObject* infer, vm::MirroredObject* compute)>&) const {}

}  // namespace eager
}  // namespace oneflow
