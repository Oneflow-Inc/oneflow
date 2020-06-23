#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/device_helper_stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/mem_instruction.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/vm/mem_buffer_object.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

namespace {

class CudaMallocInstructionType final : public InstructionType {
 public:
  CudaMallocInstructionType() = default;
  ~CudaMallocInstructionType() override = default;

  using stream_type = DeviceHelperStreamType;

  void Infer(Instruction* instruction) const override {
    MemBufferObjectType* mem_buffer_object_type = nullptr;
    size_t size = 0;
    int64_t device_id = 0;
    {
      FlatMsgView<MallocInstruction> view;
      CHECK(view.Match(instruction->instr_msg().operand()));
      size = view->size();
      RwMutexedObject* rw_mutexed_object = instruction->mut_operand_type(view->mem_buffer());
      mem_buffer_object_type = rw_mutexed_object->Init<MemBufferObjectType>();
      device_id = instruction->stream().device_id();
    }
    mem_buffer_object_type->set_size(size);
    mem_buffer_object_type->mut_mem_case()->mutable_device_cuda_mem()->set_device_id(device_id);
  }
  void Compute(Instruction* instruction) const override {
    const MemBufferObjectType* buffer_type = nullptr;
    MemBufferObjectValue* buffer_value = nullptr;
    char* dptr = nullptr;
    {
      FlatMsgView<MallocInstruction> view;
      CHECK(view.Match(instruction->instr_msg().operand()));
      const auto& operand = view->mem_buffer();
      buffer_type = &instruction->mut_operand_type(operand)->Get<MemBufferObjectType>();
      buffer_value = instruction->mut_operand_value(operand)->Init<MemBufferObjectValue>();
    }
    const auto& stream = instruction->stream();
    cudaSetDevice(stream.device_id());
    CudaCheck(cudaMalloc(&dptr, buffer_type->size()));
    buffer_value->reset_data(dptr);
  }
};
COMMAND(RegisterInstructionType<CudaMallocInstructionType>("CudaMalloc"));

class CudaFreeInstructionType final : public InstructionType {
 public:
  CudaFreeInstructionType() = default;
  ~CudaFreeInstructionType() override = default;

  using stream_type = DeviceHelperStreamType;

  void Infer(Instruction* instruction) const override {
    RwMutexedObject* type_rw_mutexed_object = nullptr;
    {
      FlatMsgView<FreeInstruction> view;
      CHECK(view.Match(instruction->instr_msg().operand()));
      type_rw_mutexed_object = instruction->mut_operand_type(view->mem_buffer());
      const auto& buffer_type = type_rw_mutexed_object->Get<MemBufferObjectType>();
      CHECK(buffer_type.mem_case().has_device_cuda_mem());
    }
    type_rw_mutexed_object->reset_object();
  }
  void Compute(Instruction* instruction) const override {
    RwMutexedObject* value_rw_mutexed_object = nullptr;
    {
      FlatMsgView<FreeInstruction> view;
      CHECK(view.Match(instruction->instr_msg().operand()));
      value_rw_mutexed_object = instruction->mut_operand_value(view->mem_buffer());
    }
    const auto& stream = instruction->stream();
    cudaSetDevice(stream.device_id());
    CudaCheck(cudaFree(value_rw_mutexed_object->Mut<MemBufferObjectValue>()->mut_data()));
    value_rw_mutexed_object->reset_object();
  }
};
COMMAND(RegisterInstructionType<CudaFreeInstructionType>("CudaFree"));

}  // namespace

void DeviceHelperStreamType::InitInstructionStatus(const Stream& stream,
                                                   InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(NaiveInstrStatusQuerier) < kInstructionStatusBufferBytes, "");
  NaiveInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data());
}

void DeviceHelperStreamType::DeleteInstructionStatus(const Stream& stream,
                                                     InstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool DeviceHelperStreamType::QueryInstructionStatusDone(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return NaiveInstrStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void DeviceHelperStreamType::Compute(Instruction* instruction) const {
  {
    const auto& instr_type_id = instruction->mut_instr_msg()->instr_type_id();
    CHECK_EQ(instr_type_id.stream_type_id().interpret_type(), InterpretType::kCompute);
    instr_type_id.instruction_type().Compute(instruction);
  }
  auto* status_buffer = instruction->mut_status_buffer();
  NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data())->set_done();
}

ObjectMsgPtr<StreamDesc> DeviceHelperStreamType::MakeStreamDesc(const Resource& resource,
                                                                int64_t this_machine_id) const {
  std::size_t device_num = 0;
  if (resource.has_cpu_device_num()) {
    device_num = std::max<std::size_t>(device_num, resource.cpu_device_num());
  }
  if (resource.has_gpu_device_num()) {
    device_num = std::max<std::size_t>(device_num, resource.gpu_device_num());
  }
  CHECK_GT(device_num, 0);
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->mutable_stream_type_id()->__Init__(LookupStreamType4TypeIndex<DeviceHelperStreamType>());
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(device_num);
  ret->set_num_streams_per_thread(1);
  ret->set_start_global_device_id(this_machine_id * device_num);
  return ret;
}

}  // namespace vm
}  // namespace oneflow
