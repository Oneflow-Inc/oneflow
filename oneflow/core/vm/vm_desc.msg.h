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
#ifndef ONEFLOW_CORE_VM_MEM_ZONE_TYPE_DESC_MSG_H_
#define ONEFLOW_CORE_VM_MEM_ZONE_TYPE_DESC_MSG_H_

#include "oneflow/core/vm/stream_desc.msg.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/interpret_type.h"
#include "oneflow/core/vm/vm_resource_desc.msg.h"
#include "oneflow/core/common/range.h"

namespace oneflow {
namespace vm {

// clang-format off
OBJECT_MSG_BEGIN(VmDesc);
  // methods
  OF_PUBLIC void __Init__(const VmResourceDesc& vm_resource_desc) {
    __Init__(vm_resource_desc, Range(0, 1));
  }
  OF_PUBLIC void __Init__(const VmResourceDesc& vm_resource_desc, const Range& machine_id_range) {
    mutable_vm_resource_desc()->CopyFrom(vm_resource_desc);
    *mutable_machine_id_range() = machine_id_range;
  }
  
  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(VmResourceDesc, vm_resource_desc);
  OBJECT_MSG_DEFINE_STRUCT(Range, machine_id_range);

  // links
  OBJECT_MSG_DEFINE_SKIPLIST_HEAD(StreamDesc, stream_type_id, stream_type_id2desc);
OBJECT_MSG_END(VmDesc);
// clang-format on

ObjectMsgPtr<VmDesc> MakeVmDesc(const Resource& resource, int64_t this_machine_id);
ObjectMsgPtr<VmDesc> MakeVmDesc(const Resource& resource, int64_t this_machine_id,
                                const std::set<std::string>& instr_type_names);

}  // namespace vm
}  // namespace oneflow

#endif  // ONEFLOW_CORE_VM_MEM_ZONE_TYPE_DESC_MSG_H_
