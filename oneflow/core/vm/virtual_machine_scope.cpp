#include "oneflow/core/vm/virtual_machine_scope.h"
#include "oneflow/core/vm/virtual_machine.msg.h"
#include "oneflow/core/vm/oneflow_vm.h"
#include "oneflow/core/job/machine_context.h"

namespace oneflow {
namespace vm {

VirtualMachineScope::VirtualMachineScope(const Resource& resource) {
  const auto& machine_ctx = *Global<MachineCtx>::Get();
  if (machine_ctx.IsThisMachineMaster()) {
    Global<OneflowVM<vm::kMaster>>::New(resource, machine_ctx.this_machine_id());
  }
  Global<OneflowVM<vm::kWorker>>::New(resource, machine_ctx.this_machine_id());
}

VirtualMachineScope::~VirtualMachineScope() {
  Global<OneflowVM<vm::kWorker>>::Delete();
  if (Global<OneflowVM<vm::kMaster>>::Get()) { Global<OneflowVM<vm::kMaster>>::Delete(); }
}

}
}
