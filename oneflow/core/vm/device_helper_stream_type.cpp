#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/mem_instruction.msg.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/vm/mem_buffer_object.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {
namespace vm {

class DeviceHelperStreamType final : public StreamType {
 public:
  DeviceHelperStreamType() = default;
  ~DeviceHelperStreamType() = default;

  void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const override {}

  void InitInstructionStatus(const Stream& stream,
                             InstructionStatusBuffer* status_buffer) const override;
  void DeleteInstructionStatus(const Stream& stream,
                               InstructionStatusBuffer* status_buffer) const override;
  bool QueryInstructionStatusDone(const Stream& stream,
                                  const InstructionStatusBuffer& status_buffer) const override;
  void Compute(InstrChain* instr_chain) const override;
  ObjectMsgPtr<StreamDesc> MakeRemoteStreamDesc(const Resource& resource,
                                                int64_t this_machine_id) const override;
};

namespace {

class CudaMallocInstructionType final : public InstructionType {
 public:
  CudaMallocInstructionType() = default;
  ~CudaMallocInstructionType() override = default;

  using stream_type = DeviceHelperStreamType;

  void Infer(InstrCtx* instr_ctx) const override {
    MemBufferObjectType* mem_buffer_object_type = nullptr;
    size_t size = 0;
    int64_t device_id = 0;
    {
      FlatMsgView<MallocInstruction> view;
      CHECK(view.Match(instr_ctx->instr_msg().operand()));
      size = view->size();
      const auto& operand = view->mirrored_object_operand().operand();
      MirroredObject* mirrored_object = instr_ctx->mut_operand_type(operand);
      mem_buffer_object_type = mirrored_object->Mutable<MemBufferObjectType>();
      device_id = instr_ctx->mut_instr_chain()->stream().thread_ctx().device_id();
    }
    mem_buffer_object_type->set_size(size);
    mem_buffer_object_type->mut_mem_case()->mutable_device_cuda_mem()->set_device_id(device_id);
  }
  void Compute(InstrCtx* instr_ctx) const override {
    const MemBufferObjectType* buffer_type = nullptr;
    MemBufferObjectValue* buffer_value = nullptr;
    char* dptr = nullptr;
    {
      FlatMsgView<MallocInstruction> view;
      CHECK(view.Match(instr_ctx->instr_msg().operand()));
      const auto& operand = view->mirrored_object_operand().operand();
      buffer_type = &instr_ctx->mut_operand_type(operand)->Get<MemBufferObjectType>();
      buffer_value = instr_ctx->mut_operand_value(operand)->Mutable<MemBufferObjectValue>();
    }
    const auto& stream = instr_ctx->mut_instr_chain()->stream();
    cudaSetDevice(stream.thread_ctx().device_id());
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

  void Infer(InstrCtx* instr_ctx) const override {
    MirroredObject* type_mirrored_object = nullptr;
    {
      FlatMsgView<FreeInstruction> view;
      CHECK(view.Match(instr_ctx->instr_msg().operand()));
      const auto& operand = view->mirrored_object_operand().operand();
      type_mirrored_object = instr_ctx->mut_operand_type(operand);
      const auto& buffer_type = type_mirrored_object->Get<MemBufferObjectType>();
      CHECK(buffer_type.mem_case().has_device_cuda_mem());
    }
    type_mirrored_object->reset_object();
  }
  void Compute(InstrCtx* instr_ctx) const override {
    MirroredObject* value_mirrored_object = nullptr;
    {
      FlatMsgView<FreeInstruction> view;
      CHECK(view.Match(instr_ctx->instr_msg().operand()));
      const auto& operand = view->mirrored_object_operand().operand();
      value_mirrored_object = instr_ctx->mut_operand_value(operand);
    }
    const auto& stream = instr_ctx->mut_instr_chain()->stream();
    cudaSetDevice(stream.thread_ctx().device_id());
    CudaCheck(cudaFree(value_mirrored_object->Mut<MemBufferObjectValue>()->mut_data()));
    value_mirrored_object->reset_object();
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

void DeviceHelperStreamType::Compute(InstrChain* instr_chain) const {
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(instr_chain->mut_instr_ctx_list(), instr_ctx) {
    const auto& instr_type_id = instr_ctx->mut_instr_msg()->instr_type_id();
    CHECK_EQ(instr_type_id.stream_type_id().interpret_type(), InterpretType::kCompute);
    instr_type_id.instruction_type().Compute(instr_ctx);
  }
  auto* status_buffer = instr_chain->mut_status_buffer();
  NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data())->set_done();
}

ObjectMsgPtr<StreamDesc> DeviceHelperStreamType::MakeRemoteStreamDesc(
    const Resource& resource, int64_t this_machine_id) const {
  std::size_t device_num = 0;
  if (resource.has_cpu_device_num()) {
    device_num = resource.cpu_device_num();
  } else if (resource.has_gpu_device_num()) {
    device_num = resource.gpu_device_num();
  } else {
    UNIMPLEMENTED();
  }
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->mutable_stream_type_id()->__Init__(LookupStreamType4TypeIndex<DeviceHelperStreamType>());
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(device_num);
  ret->set_num_streams_per_thread(1);
  ret->set_start_parallel_id(this_machine_id * device_num);
  return ret;
}

}  // namespace vm
}  // namespace oneflow
