#include "oneflow/core/vm/vm_stream.msg.h"
#include "oneflow/core/vm/vm_thread.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

void VmStream::__Init__(VmThread* vm_thread, const VmStreamId& vm_stream_id) {
  set_vm_thread(vm_thread);
  mut_vm_stream_id()->CopyFrom(vm_stream_id);
  const auto& vm_stream_type = vm_thread->vm_stream_rt_desc().vm_stream_type();
  vm_stream_type.InitDeviceCtx(mut_device_ctx(), this);
}

int64_t VmStream::machine_id() const {
  return parallel_id() / vm_thread().vm_stream_rt_desc().vm_stream_desc().num_streams_per_machine();
}

ObjectMsgPtr<VmInstrChain> VmStream::NewVmInstrChain(VmInstructionMsg* vm_instr_msg) {
  if (free_chain_list().empty()) {
    return ObjectMsgPtr<VmInstrChain>::NewFrom(mut_allocator(), vm_instr_msg, this);
  }
  ObjectMsgPtr<VmInstrChain> vm_instr_chain = mut_free_chain_list()->PopFront();
  vm_instr_chain->__Init__(vm_instr_msg, this);
  return vm_instr_chain;
}

void VmStream::DeleteVmInstrChain(ObjectMsgPtr<VmInstrChain>&& vm_instr_chain) {
  CHECK(vm_instr_chain->is_pending_chain_link_empty());
  CHECK(vm_instr_chain->is_vm_instr_chain_link_empty());
  CHECK_EQ(vm_instr_chain->ref_cnt(), 1);
  auto* vm_instr_chain_ptr = vm_instr_chain.Mutable();
  mut_free_chain_list()->EmplaceBack(std::move(vm_instr_chain));
  vm_instr_chain_ptr->__Delete__();
}

}  // namespace oneflow
