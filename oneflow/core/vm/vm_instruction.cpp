#include "oneflow/core/vm/vm_instruction.msg.h"
#include "oneflow/core/vm/vm_stream_type.h"
#include "oneflow/core/vm/vm_stream.msg.h"
#include "oneflow/core/vm/vm_thread.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

void VmInstrChain::__Init__(VmInstructionMsg* vm_instr_msg, VmStream* vm_stream) {
  mutable_status_buffer();
  set_vm_stream(vm_stream);
  set_vm_stream_type(&vm_stream->vm_thread().vm_stream_rt_desc().vm_stream_type());
  vm_stream_type().InitVmInstructionStatus(*vm_stream, mutable_status_buffer());
  auto vm_instruction = ObjectMsgPtr<VmInstruction>::NewFrom(mut_allocator(), this, vm_instr_msg);
  mut_vm_instruction_list()->EmplaceBack(std::move(vm_instruction));
  CHECK_EQ(vm_instruction_list().size(), 1);
}

void VmInstrChain::__Delete__() {
  vm_stream_type().DeleteVmInstructionStatus(vm_stream(), mut_status_buffer());
}

bool VmInstrChain::Done() const {
  return vm_stream_type().QueryVmInstructionStatusDone(vm_stream(), status_buffer());
}

void VmInstrChainPackage::__Init__(VmStream* vm_stream) {
  mutable_status_buffer();
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

}  // namespace oneflow
