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
#include "oneflow/core/common/decorator.h"
#include "oneflow/core/eager/lazy_job_phy_instr_operand.h"
#include "oneflow/core/common/container_util.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/stream.h"
#include "oneflow/core/vm/virtual_machine.h"

namespace oneflow {
namespace vm {

void LaunchLazyJobPhyInstrOperand::ForEachMutMirroredObject(
    const std::function<void(vm::MirroredObject* compute)>& DoEach) const {
  for (const auto& eager_blob_object : *param_blob_objects_) {
    DoEach(CHECK_JUST(eager_blob_object->compute_local_dep_object()));
  }
  DoEach(CHECK_JUST(SingletonMaybe<VirtualMachine>())
             ->FindOrCreateTransportLocalDepObject()
             .Mutable());
}

}  // namespace vm
}  // namespace oneflow
