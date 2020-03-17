#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/thread.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

InstructionOperand* InstructionMsg::add_operand() {
  mut_operand()->emplace_back();
  return mut_operand()->back().Mutable();
}

void InstructionMsg::__Init__(const InstructionProto& proto) {
  mutable_vm_instr_id()->__Init__(proto.instr_type_name());
  mutable_operand()->resize(proto.operand_size());
  for (int i = 0; i < proto.operand_size(); ++i) {
    mutable_operand()->at(i)->__Init__(proto.operand(i));
  }
}

MirroredObject* Instruction::FindMirroredObjectByOperand(const MirroredObjectOperand& operand,
                                                         int64_t default_parallel_id) {
  FlatMsg<MirroredObjectId> mirrored_object_id;
  mirrored_object_id->__Init__(operand, default_parallel_id);
  auto* access = mut_mirrored_object_id2access()->FindPtr(mirrored_object_id.Get());
  if (access == nullptr) { return nullptr; }
  return access->mut_mirrored_object();
}

void InstrChain::__Init__(InstructionMsg* vm_instr_msg, Stream* vm_stream) {
  mutable_status_buffer();
  set_vm_stream(vm_stream);
  set_vm_stream_type(&vm_stream->vm_thread().vm_stream_rt_desc().vm_stream_type());
  vm_stream_type().InitInstructionStatus(*vm_stream, mutable_status_buffer());
  auto vm_instruction = ObjectMsgPtr<Instruction>::NewFrom(mut_allocator(), this, vm_instr_msg);
  mut_vm_instruction_list()->EmplaceBack(std::move(vm_instruction));
  CHECK_EQ(vm_instruction_list().size(), 1);
}

void InstrChain::__Delete__() {
  vm_stream_type().DeleteInstructionStatus(vm_stream(), mut_status_buffer());
  mut_vm_instruction_list()->Clear();
  mut_in_edges()->Clear();
  mut_out_edges()->Clear();
}

bool InstrChain::Done() const {
  return vm_stream_type().QueryInstructionStatusDone(vm_stream(), status_buffer());
}

}  // namespace vm
}  // namespace oneflow
