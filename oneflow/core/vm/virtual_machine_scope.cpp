#include "oneflow/core/vm/virtual_machine_scope.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/oneflow_vm.h"
#include "oneflow/core/job/machine_context.h"

namespace oneflow {
namespace vm {

VirtualMachineScope::VirtualMachineScope(const Resource& resource) {
  const auto& machine_ctx = *Global<MachineCtx>::Get();
  Global<OneflowVM>::New(resource, machine_ctx.this_machine_id());
}

VirtualMachineScope::~VirtualMachineScope() { Global<OneflowVM>::Delete(); }

}  // namespace vm
}  // namespace oneflow
