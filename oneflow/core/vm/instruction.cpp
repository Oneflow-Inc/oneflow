#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

InstructionOperand* InstructionMsg::add_operand() {
  mut_operand()->emplace_back();
  return mut_operand()->back().Mutable();
}

void InstructionMsg::__Init__(const InstructionProto& proto) {
  mutable_instr_type_id()->__Init__(proto.instr_type_name());
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

void InstrChain::__Init__(InstructionMsg* instr_msg, Stream* stream) {
  mutable_status_buffer();
  set_stream(stream);
  set_stream_type(&stream->thread_ctx().stream_rt_desc().stream_type());
  stream_type().InitInstructionStatus(*stream, mutable_status_buffer());
  auto instruction = ObjectMsgPtr<Instruction>::NewFrom(mut_allocator(), this, instr_msg);
  mut_instruction_list()->EmplaceBack(std::move(instruction));
  CHECK_EQ(instruction_list().size(), 1);
}

void InstrChain::__Delete__() {
  stream_type().DeleteInstructionStatus(stream(), mut_status_buffer());
  mut_instruction_list()->Clear();
  mut_in_edges()->Clear();
  mut_out_edges()->Clear();
}

bool InstrChain::Done() const {
  return stream_type().QueryInstructionStatusDone(stream(), status_buffer());
}

}  // namespace vm
}  // namespace oneflow
