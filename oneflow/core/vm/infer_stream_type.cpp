#include "oneflow/core/vm/infer_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"

namespace oneflow {
namespace vm {

void InferStreamTypeUtil::InitInstructionStatus(const Stream& stream,
                                                InstructionStatusBuffer* status_buffer) {
  static_assert(sizeof(NaiveInstrStatusQuerier) < kInstructionStatusBufferBytes, "");
  NaiveInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data());
}

void InferStreamTypeUtil::DeleteInstructionStatus(const Stream& stream,
                                                  InstructionStatusBuffer* status_buffer) {
  // do nothing
}

bool InferStreamTypeUtil::QueryInstructionStatusDone(const Stream& stream,
                                                     const InstructionStatusBuffer& status_buffer) {
  return NaiveInstrStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void InferStreamTypeUtil::Infer(Instruction* instruction) {
  {
    const auto& instr_type_id = instruction->mut_instr_msg()->instr_type_id();
    CHECK_EQ(instr_type_id.stream_type_id().interpret_type(), InterpretType::kInfer);
    instr_type_id.instruction_type().Infer(instruction);
  }
  auto* status_buffer = instruction->mut_status_buffer();
  NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data())->set_done();
}

}  // namespace vm
}  // namespace oneflow
