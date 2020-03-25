#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
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

enum DeviceHelperInstrOpCode { kCudaMallocOpcode = 0, kCudaFreeOpcode };

class CudaMallocInstructionType final : public InstructionType {
 public:
  CudaMallocInstructionType() = default;
  ~CudaMallocInstructionType() override = default;

  using stream_type = DeviceHelperStreamType;
  static const InstructionOpcode opcode = kCudaMallocOpcode;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(CudaMallocInstruction);
    FLAT_MSG_VIEW_DEFINE_PATTERN(MutableMirroredObjectOperand, mirrored_object_operand);
    FLAT_MSG_VIEW_DEFINE_PATTERN(uint64_t, size);
  FLAT_MSG_VIEW_END(CudaMallocInstruction);
  // clang-format on

  void Compute(Instruction* instr) const override {
    MirroredObject* mirrored_object = nullptr;
    char* dptr = nullptr;
    size_t size = 0;
    const auto& stream = instr->mut_instr_chain()->stream();
    {
      auto parallel_num = stream.thread_ctx().stream_rt_desc().stream_desc().parallel_num();
      FlatMsgView<CudaMallocInstruction> view;
      CHECK(view->Match(instr->mut_instr_msg()->mut_operand()));
      size = view->size();
      FlatMsg<MirroredObjectId> mirrored_object_id;
      mirrored_object_id->__Init__(view->mirrored_object_operand().operand(), stream.parallel_id());
      auto* mirrored_object_access =
          instr->mut_mirrored_object_id2access()->FindPtr(mirrored_object_id.Get());
      CHECK_NOTNULL(mirrored_object_access);
      mirrored_object = mirrored_object_access->mut_mirrored_object();
      CHECK_EQ(mirrored_object->parallel_id(), stream.parallel_id());
      CHECK_EQ(mirrored_object->logical_object().parallel_id2mirrored_object().size(),
               parallel_num);
      CHECK(!mirrored_object->has_object_type());
    }
    {
      cudaSetDevice(stream.thread_ctx().device_id());
      CudaCheck(cudaMalloc(&dptr, size));
    }
    mirrored_object->mutable_cuda_mem_buffer()->__Init__(size, dptr);
  }
};
COMMAND(RegisterInstrTypeId<CudaMallocInstructionType>("CudaMalloc", kRemote));

class CudaFreeInstructionType final : public InstructionType {
 public:
  CudaFreeInstructionType() = default;
  ~CudaFreeInstructionType() override = default;

  using stream_type = DeviceHelperStreamType;
  static const InstructionOpcode opcode = kCudaFreeOpcode;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(CudaFreeInstruction);
    FLAT_MSG_VIEW_DEFINE_PATTERN(MutableMirroredObjectOperand, mirrored_object_operand);
  FLAT_MSG_VIEW_END(CudaFreeInstruction);
  // clang-format on

  void Compute(Instruction* instr) const override {
    MirroredObject* mirrored_object = nullptr;
    const auto& stream = instr->mut_instr_chain()->stream();
    {
      auto parallel_num = stream.thread_ctx().stream_rt_desc().stream_desc().parallel_num();
      FlatMsgView<CudaFreeInstruction> view;
      CHECK(view->Match(instr->mut_instr_msg()->mut_operand()));
      FlatMsg<MirroredObjectId> mirrored_object_id;
      mirrored_object_id->__Init__(view->mirrored_object_operand().operand(), stream.parallel_id());
      auto* mirrored_object_access =
          instr->mut_mirrored_object_id2access()->FindPtr(mirrored_object_id.Get());
      CHECK_NOTNULL(mirrored_object_access);
      mirrored_object = mirrored_object_access->mut_mirrored_object();
      CHECK_EQ(mirrored_object->parallel_id(), stream.parallel_id());
      CHECK_EQ(mirrored_object->logical_object().parallel_id2mirrored_object().size(),
               parallel_num);
    }
    {
      cudaSetDevice(stream.thread_ctx().device_id());
      CudaCheck(cudaFree(mirrored_object->mut_cuda_mem_buffer()->mut_data()));
    }
    mirrored_object->clear_cuda_mem_buffer();
  }
};
COMMAND(RegisterInstrTypeId<CudaFreeInstructionType>("CudaFree", kRemote));

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
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(instr_chain->mut_instruction_list(), instruction) {
    const auto& instr_type_id = instruction->mut_instr_msg()->instr_type_id();
    CHECK_EQ(instr_type_id.stream_type_id().interpret_type(), InterpretType::kCompute);
    LookupInstructionType(instr_type_id)->Compute(instruction);
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
  ret->mutable_stream_type_id()->__Init__(typeid(DeviceHelperStreamType));
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(device_num);
  ret->set_num_streams_per_thread(1);
  ret->set_start_parallel_id(this_machine_id * device_num);
  return ret;
}

COMMAND(RegisterStreamType<DeviceHelperStreamType>());

}  // namespace vm
}  // namespace oneflow
