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
#ifndef ONEFLOW_CORE_FRAMEWORK_LOCAL_DEP_OBJECT_H_
#define ONEFLOW_CORE_FRAMEWORK_LOCAL_DEP_OBJECT_H_

#include "oneflow/core/object_msg/object_msg_core.h"
#include "oneflow/core/vm/vm_object.msg.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/symbol.h"

namespace oneflow {

class Device;

// clang-format off

// Helps VirtualMachine building instruction edges
OBJECT_MSG_BEGIN(LocalDepObject);
  // methods
  OF_PUBLIC Maybe<void> Init(const Device& device);

  // fields
  OBJECT_MSG_DEFINE_OPTIONAL(vm::LogicalObject, logical_object);
  OBJECT_MSG_DEFINE_OPTIONAL(vm::MirroredObject, mirrored_object);

  // links
  OBJECT_MSG_DEFINE_LIST_LINK(free_link);
  OBJECT_MSG_DEFINE_LIST_LINK(zombie_link);
OBJECT_MSG_END(LocalDepObject);
// clang-format on

Maybe<LocalDepObject*> GetLocalDepObject(Symbol<Device> device);
Maybe<LocalDepObject*> FindOrCreateComputeLocalDepObject(const Device& device);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_LOCAL_DEP_OBJECT_H_
