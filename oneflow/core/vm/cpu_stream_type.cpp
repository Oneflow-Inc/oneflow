#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/cpu_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/mem_instruction.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/vm/mem_buffer_object.h"
#include "oneflow/core/device/cpu_device_context.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

void CpuStreamType::InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const {
  device_ctx->reset(new CpuDeviceCtx());
}

void CpuStreamType::InitInstructionStatus(const Stream& stream,
                                          InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(NaiveInstrStatusQuerier) < kInstructionStatusBufferBytes, "");
  NaiveInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data());
}  // namespace
   // voidCpuStreamType::InitInstructionStatus(constStream&stream,InstructionStatusBuffer*status_buffer)const

void CpuStreamType::DeleteInstructionStatus(const Stream& stream,
                                            InstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool CpuStreamType::QueryInstructionStatusDone(const Stream& stream,
                                               const InstructionStatusBuffer& status_buffer) const {
  return NaiveInstrStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void CpuStreamType::Compute(Instruction* instruction) const {
  {
    const auto& instr_type_id = instruction->mut_instr_msg()->instr_type_id();
    CHECK_EQ(instr_type_id.stream_type_id().interpret_type(), InterpretType::kCompute);
    instr_type_id.instruction_type().Compute(instruction);
  }
  auto* status_buffer = instruction->mut_status_buffer();
  NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data())->set_done();
}

ObjectMsgPtr<StreamDesc> CpuStreamType::MakeStreamDesc(const Resource& resource,
                                                       int64_t this_machine_id) const {
  if (!resource.has_cpu_device_num()) { return ObjectMsgPtr<StreamDesc>(); }
  std::size_t device_num = resource.cpu_device_num();
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->mutable_stream_type_id()->__Init__(LookupStreamType4TypeIndex<CpuStreamType>());
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(device_num);
  ret->set_num_streams_per_thread(1);
  ret->set_start_global_device_id(this_machine_id * device_num);
  return ret;
}

}  // namespace vm
}  // namespace oneflow
