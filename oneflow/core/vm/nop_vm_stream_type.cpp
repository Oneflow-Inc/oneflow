#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/nop_vm_stream_type.h"
#include "oneflow/core/vm/vm_instruction.msg.h"
#include "oneflow/core/vm/vm_stream.msg.h"
#include "oneflow/core/vm/vm_thread.msg.h"
#include "oneflow/core/vm/naive_vm_instruction_status_querier.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace {}  // namespace

void NopVmStreamType::InitVmInstructionStatus(const VmStream& vm_stream,
                                              VmInstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(NaiveVmInstrStatusQuerier) < kVmInstructionStatusBufferBytes, "");
  NaiveVmInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data());
}

void NopVmStreamType::DeleteVmInstructionStatus(const VmStream& vm_stream,
                                                VmInstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool NopVmStreamType::QueryVmInstructionStatusDone(
    const VmStream& vm_stream, const VmInstructionStatusBuffer& status_buffer) const {
  return NaiveVmInstrStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

const VmStreamTypeId NopVmStreamType::kVmStreamTypeId;

ObjectMsgPtr<VmInstructionMsg> NopVmStreamType::Nop() const {
  auto vm_instr_msg = ObjectMsgPtr<VmInstructionMsg>::New();
  auto* vm_instr_proto = vm_instr_msg->mutable_vm_instruction_proto();
  vm_instr_proto->set_vm_stream_type_id(kVmStreamTypeId);
  vm_instr_proto->set_opcode(0);
  vm_instr_proto->mutable_vm_stream_mask()->mutable_all_vm_stream_enabled();
  return vm_instr_msg;
}

void NopVmStreamType::Run(VmInstrChain* vm_instr_chain) const {
  auto* status_buffer = vm_instr_chain->mut_status_buffer();
  NaiveVmInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data())->set_done();
}

COMMAND(RegisterVmStreamType<NopVmStreamType>());

}  // namespace oneflow
