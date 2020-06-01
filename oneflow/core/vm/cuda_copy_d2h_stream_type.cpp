#include "oneflow/core/vm/cuda_copy_d2h_stream_type.h"
#include "oneflow/core/vm/cuda_copy_d2h_device_context.h"

namespace oneflow {
namespace vm {

void CudaCopyD2HStreamType::InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx,
                                          Stream* stream) const {
  device_ctx->reset(new CudaCopyD2HDeviceCtx(stream->mut_callback_list()));
}

void CudaCopyD2HStreamType::InitInstructionStatus(const Stream& stream,
                                                  InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(CudaInstrStatusQuerier) < kInstructionStatusBufferBytes, "");
  CudaInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data(), stream.device_id());
}

void CudaCopyD2HStreamType::DeleteInstructionStatus(const Stream& stream,
                                                    InstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool CudaCopyD2HStreamType::QueryInstructionStatusDone(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return CudaInstrStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void CudaCopyD2HStreamType::Compute(Instruction* instruction) const {
  auto* stream = instruction->mut_stream();
  cudaSetDevice(stream->device_id());
  {
    const auto& instr_type_id = instruction->mut_instr_msg()->instr_type_id();
    CHECK_EQ(instr_type_id.stream_type_id().interpret_type(), InterpretType::kCompute);
    instr_type_id.instruction_type().Compute(instruction);
  }
  stream->mut_callback_list()->MoveTo(instruction->mut_callback_list());
  char* data_ptr = instruction->mut_status_buffer()->mut_buffer()->mut_data();
  CudaInstrStatusQuerier::MutCast(data_ptr)->SetLaunched(stream->device_ctx().get());
}

ObjectMsgPtr<StreamDesc> CudaCopyD2HStreamType::MakeStreamDesc(const Resource& resource,
                                                               int64_t this_machine_id) const {
  std::size_t device_num = resource.gpu_device_num();
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->mutable_stream_type_id()->__Init__(LookupStreamType4TypeIndex<CudaCopyD2HStreamType>());
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(device_num);
  ret->set_num_streams_per_thread(1);
  ret->set_start_global_device_id(this_machine_id * device_num);
  return ret;
}

}  // namespace vm
}  // namespace oneflow
