#include "oneflow/core/vm/vm_instruction.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

void VmStreamRtDesc::__Init__(const VmStreamDesc* vm_stream_desc) {
  VmStreamTypeId vm_stream_type_id = vm_stream_desc->vm_stream_type_id();
  const VmStreamType* vm_stream_type = LookupVmStreamType(vm_stream_type_id);
  set_vm_stream_type(vm_stream_type);
  set_vm_stream_desc(vm_stream_desc);
  set_vm_stream_type_id(vm_stream_type_id);
}

void VmInstrChain::__Init__(VmInstructionMsg* vm_instruction_msg, VmStream* vm_stream) {
  set_vm_stream(vm_stream);
  auto vm_instruction = ObjectMsgPtr<VmInstruction>::NewFrom(mut_allocator(), vm_instruction_msg);
  mut_vm_instruction_list()->EmplaceBack(std::move(vm_instruction));
}

void VmInstrChainPackage::__Init__(VmStream* vm_stream) {
  set_vm_stream(vm_stream);
  set_vm_stream_type(&vm_stream->vm_thread().vm_stream_rt_desc().vm_stream_type());
  vm_stream_type().InitVmInstructionStatus(*vm_stream, mutable_status_buffer());
}

void VmInstrChainPackage::__Delete__() {
  vm_stream_type().DeleteVmInstructionStatus(vm_stream(), mut_status_buffer());
}

bool VmInstrChainPackage::Done() const {
  return vm_stream_type().QueryVmInstructionStatusDone(vm_stream(), status_buffer());
}

ObjectMsgPtr<VmInstrChainPackage> VmStream::NewVmInstrChainPackage() {
  if (free_pkg_list().empty()) {
    return ObjectMsgPtr<VmInstrChainPackage>::NewFrom(mut_allocator(), this);
  }
  ObjectMsgPtr<VmInstrChainPackage> vm_instr_chain_pkg = mut_free_pkg_list()->PopFront();
  vm_instr_chain_pkg->__Init__(this);
  return vm_instr_chain_pkg;
}

void VmStream::DeleteVmInstrChainPackage(VmInstrChainPackage* vm_instr_chain_pkg) {
  CHECK(vm_instr_chain_pkg->is_waiting_pkg_link_empty());
  mut_running_pkg_list()->MoveToDstBack(vm_instr_chain_pkg, mut_free_pkg_list());
  vm_instr_chain_pkg->__Delete__();
}

}  // namespace oneflow
