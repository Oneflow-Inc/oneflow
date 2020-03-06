#include "oneflow/core/vm/vm_stream.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

ObjectMsgPtr<VmInstrChainPackage> VmStream::NewVmInstrChainPackage() {
  if (free_pkg_list().empty()) {
    return ObjectMsgPtr<VmInstrChainPackage>::NewFrom(mut_allocator(), this);
  }
  ObjectMsgPtr<VmInstrChainPackage> vm_instr_chain_pkg = mut_free_pkg_list()->PopFront();
  vm_instr_chain_pkg->__Init__(this);
  return vm_instr_chain_pkg;
}

void VmStream::DeleteVmInstrChainPackage(VmInstrChainPackage* vm_instr_chain_pkg) {
  CHECK(vm_instr_chain_pkg->is_pending_pkg_link_empty());
  mut_running_pkg_list()->MoveToDstBack(vm_instr_chain_pkg, mut_free_pkg_list());
  vm_instr_chain_pkg->__Delete__();
}

}  // namespace oneflow
