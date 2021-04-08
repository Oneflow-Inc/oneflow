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
#include "oneflow/core/framework/vm_local_dep_object.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/vm/id_util.h"
#include "oneflow/core/vm/vm_object.msg.h"
#include "oneflow/core/vm/oneflow_vm.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/control/global_process_ctx.h"

namespace oneflow {
namespace one {

namespace {

void InitLocalObject(int64_t global_device_id, const std::shared_ptr<ParallelDesc>& parallel_desc,
                     const vm::ObjectId& object_id, ObjectMsgPtr<vm::LogicalObject>* logical_object,
                     ObjectMsgPtr<vm::MirroredObject>* mirrored_object) {
  *logical_object = ObjectMsgPtr<vm::LogicalObject>::New(object_id, parallel_desc);
  *mirrored_object =
      ObjectMsgPtr<vm::MirroredObject>::New(logical_object->Mutable(), global_device_id);
}

}  // namespace

VmLocalDepObject::VmLocalDepObject(const std::shared_ptr<const ParallelDesc>& parallel_desc) {
  vm::ObjectId object_id = vm::IdUtil::NewPhysicalValueObjectId(GlobalProcessCtx::Rank());
  int64_t global_device_id = 0;
  {
    CHECK_EQ(parallel_desc->parallel_num(), 1);
    int64_t machine_id = CHECK_JUST(parallel_desc->MachineId4ParallelId(0));
    CHECK_EQ(machine_id, GlobalProcessCtx::Rank());
    int64_t device_id = CHECK_JUST(parallel_desc->DeviceId4ParallelId(0));
    const auto& vm = Global<OneflowVM>::Get()->vm();
    CHECK_EQ(vm.this_machine_id(), machine_id);
    global_device_id = vm.this_start_global_device_id() + device_id;
  }
  const auto& mut_parallel_desc = std::const_pointer_cast<ParallelDesc>(parallel_desc);
  InitLocalObject(global_device_id, mut_parallel_desc, object_id, &compute_logical_object_,
                  &compute_mirrored_object_);
  vm::ObjectId type_object_id = vm::IdUtil::GetTypeId(object_id);
  InitLocalObject(global_device_id, mut_parallel_desc, type_object_id, &infer_logical_object_,
                  &infer_mirrored_object_);
}

}  // namespace one
}  // namespace oneflow
