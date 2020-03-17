#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/nop_stream_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/thread.msg.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

namespace {}  // namespace

void NopStreamType::InitInstructionStatus(const Stream& vm_stream,
                                          InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(NaiveInstrStatusQuerier) < kInstructionStatusBufferBytes, "");
  NaiveInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data());
}

void NopStreamType::DeleteInstructionStatus(const Stream& vm_stream,
                                            InstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool NopStreamType::QueryInstructionStatusDone(const Stream& vm_stream,
                                               const InstructionStatusBuffer& status_buffer) const {
  return NaiveInstrStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

const StreamTypeId NopStreamType::kStreamTypeId;

ObjectMsgPtr<InstructionMsg> NopStreamType::Nop() const {
  auto vm_instr_msg = ObjectMsgPtr<InstructionMsg>::New();
  auto* vm_instr_id = vm_instr_msg->mutable_vm_instr_id();
  vm_instr_id->set_vm_stream_type_id(kStreamTypeId);
  vm_instr_id->set_opcode(0);
  return vm_instr_msg;
}

void NopStreamType::Run(InstrChain* vm_instr_chain) const {
  auto* status_buffer = vm_instr_chain->mut_status_buffer();
  NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data())->set_done();
}

COMMAND(RegisterStreamType<NopStreamType>());
COMMAND(RegisterInstructionId<NopStreamType>("Nop", 0, kRemote));
COMMAND(RegisterInstructionId<NopStreamType>("LocalNop", 0, kLocal));

}  // namespace vm
}  // namespace oneflow
