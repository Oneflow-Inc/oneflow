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
#include "oneflow/core/vm/virtual_machine_scope.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/oneflow_vm.h"
#include "oneflow/core/control/global_process_ctx.h"

namespace oneflow {
namespace vm {

VirtualMachineScope::VirtualMachineScope(const Resource& resource) {
  Global<OneflowVM>::New(resource, GlobalProcessCtx::Rank());
}

VirtualMachineScope::~VirtualMachineScope() { Global<OneflowVM>::Delete(); }

}  // namespace vm
}  // namespace oneflow
