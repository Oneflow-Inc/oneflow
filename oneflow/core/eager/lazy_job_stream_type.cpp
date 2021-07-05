#include "oneflow/core/eager/lazy_job_stream_type.h"
#include "oneflow/core/eager/lazy_job_device_ctx.h"

namespace oneflow {
namespace vm {

void LazyJobStreamType::InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const {
  device_ctx->reset(new LazyJobDeviceCtx());
}

void LazyJobStreamType::InitInstructionStatus(const Stream& stream,
                                          InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(NaiveInstrStatusQuerier) < kInstructionStatusBufferBytes, "");
  NaiveInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data());
}

void LazyJobStreamType::DeleteInstructionStatus(const Stream& stream,
                                            InstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool LazyJobStreamType::QueryInstructionStatusDone(const Stream& stream,
                                               const InstructionStatusBuffer& status_buffer) const {
  return NaiveInstrStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void LazyJobStreamType::Compute(Instruction* instruction) const {
  {
    const auto& instr_type_id = instruction->mut_instr_msg()->instr_type_id();
    CHECK_EQ(instr_type_id.stream_type_id().interpret_type(), InterpretType::kCompute);
    instr_type_id.instruction_type().Compute(instruction);
  }
}

ObjectMsgPtr<StreamDesc> LazyJobStreamType::MakeStreamDesc(const Resource& resource,
                                                       int64_t this_machine_id) const {
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->mutable_stream_type_id()->__Init__(LookupStreamType4TypeIndex<HostStreamType>());
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(1);
  ret->set_num_streams_per_thread(1);
  return ret;
}

}
}
