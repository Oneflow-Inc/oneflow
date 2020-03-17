#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/device_helper_stream_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {
namespace vm {

namespace {

enum DeviceHelperInstrOpCode { kCudaMallocOpcode = 0, kCudaFreeOpcode };

typedef void (*DeviceHelperInstrFunc)(Instruction*);
std::vector<DeviceHelperInstrFunc> device_helper_instr_table;

#define REGISTER_DEVICE_HELPER_INSTRUCTION(op_code, function_name) \
  COMMAND({                                                        \
    device_helper_instr_table.resize(op_code + 1);                 \
    device_helper_instr_table.at(op_code) = &function_name;        \
  })

// clang-format off
FLAT_MSG_VIEW_BEGIN(CudaMallocInstruction);
  FLAT_MSG_VIEW_DEFINE_PATTERN(MutableMirroredObjectOperand, mirrored_object_operand);
  FLAT_MSG_VIEW_DEFINE_PATTERN(uint64_t, size);
FLAT_MSG_VIEW_END(CudaMallocInstruction);
// clang-format on

void CudaMalloc(Instruction* instr) {
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
    CHECK_EQ(mirrored_object->logical_object().parallel_id2mirrored_object().size(), parallel_num);
    CHECK(!mirrored_object->has_object_type());
  }
  {
    cudaSetDevice(stream.thread_ctx().device_id());
    CudaCheck(cudaMalloc(&dptr, size));
  }
  mirrored_object->mutable_cuda_mem_buffer()->__Init__(size, dptr);
}
REGISTER_DEVICE_HELPER_INSTRUCTION(kCudaMallocOpcode, CudaMalloc);
COMMAND(RegisterInstrTypeId<DeviceHelperStreamType>("CudaMalloc", kCudaMallocOpcode, kRemote));

// clang-format off
FLAT_MSG_VIEW_BEGIN(CudaFreeInstruction);
  FLAT_MSG_VIEW_DEFINE_PATTERN(MutableMirroredObjectOperand, mirrored_object_operand);
FLAT_MSG_VIEW_END(CudaFreeInstruction);
// clang-format on

void CudaFree(Instruction* instr) {
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
    CHECK_EQ(mirrored_object->logical_object().parallel_id2mirrored_object().size(), parallel_num);
  }
  {
    cudaSetDevice(stream.thread_ctx().device_id());
    CudaCheck(cudaFree(mirrored_object->mut_cuda_mem_buffer()->mut_data()));
  }
  mirrored_object->clear_cuda_mem_buffer();
}
REGISTER_DEVICE_HELPER_INSTRUCTION(kCudaFreeOpcode, CudaFree);
COMMAND(RegisterInstrTypeId<DeviceHelperStreamType>("CudaFree", kCudaFreeOpcode, kRemote));

}  // namespace

const StreamTypeId DeviceHelperStreamType::kStreamTypeId;

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

ObjectMsgPtr<InstructionMsg> DeviceHelperStreamType::CudaMalloc(uint64_t logical_object_id,
                                                                size_t size) const {
  auto instr_msg = ObjectMsgPtr<InstructionMsg>::New();
  auto* instr_type_id = instr_msg->mutable_instr_type_id();
  instr_type_id->set_stream_type_id(kStreamTypeId);
  instr_type_id->set_opcode(DeviceHelperInstrOpCode::kCudaMallocOpcode);
  {
    FlatMsgView<CudaMallocInstruction> view(instr_msg->mutable_operand());
    view->mutable_mirrored_object_operand()->mutable_operand()->__Init__(logical_object_id);
    view->set_size(size);
  }
  return instr_msg;
}

ObjectMsgPtr<InstructionMsg> DeviceHelperStreamType::CudaFree(uint64_t logical_object_id) const {
  auto instr_msg = ObjectMsgPtr<InstructionMsg>::New();
  auto* instr_type_id = instr_msg->mutable_instr_type_id();
  instr_type_id->set_stream_type_id(kStreamTypeId);
  instr_type_id->set_opcode(DeviceHelperInstrOpCode::kCudaFreeOpcode);
  {
    FlatMsgView<CudaFreeInstruction> view(instr_msg->mutable_operand());
    view->mutable_mirrored_object_operand()->mutable_operand()->__Init__(logical_object_id);
  }
  return instr_msg;
}

void DeviceHelperStreamType::Run(InstrChain* instr_chain) const {
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(instr_chain->mut_instruction_list(), instruction) {
    auto opcode = instruction->mut_instr_msg()->instr_type_id().opcode();
    device_helper_instr_table.at(opcode)(instruction);
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
  ret->set_stream_type_id(kStreamTypeId);
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(device_num);
  ret->set_num_streams_per_thread(1);
  ret->set_start_parallel_id(this_machine_id * device_num);
  return ret;
}

COMMAND(RegisterStreamType<DeviceHelperStreamType>());

}  // namespace vm
}  // namespace oneflow
