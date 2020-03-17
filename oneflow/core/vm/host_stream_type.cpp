#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/host_stream_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/thread.msg.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

namespace {

enum HostInstrOpCode {
  kCudaMallocHostOpcode = 0,
  kCudaFreeHostOpcode,
  kMallocOpcode,
  kFreeOpcode,
};

typedef void (*HostInstrFunc)(Instruction*);
std::vector<HostInstrFunc> host_instr_table;

#define REGISTER_HOST_INSTRUCTION(op_code, function_name)                            \
  COMMAND({                                                                          \
    host_instr_table.resize(std::max<size_t>(host_instr_table.size(), op_code + 1)); \
    host_instr_table.at(op_code) = &function_name;                                   \
  })

// clang-format off
FLAT_MSG_VIEW_BEGIN(CudaMallocHostInstruction);
  FLAT_MSG_VIEW_DEFINE_PATTERN(MutableMirroredObjectOperand, mirrored_object_operand);
  FLAT_MSG_VIEW_DEFINE_PATTERN(uint64_t, size);
FLAT_MSG_VIEW_END(CudaMallocHostInstruction);
// clang-format on

void CudaMallocHost(Instruction* instr) {
  MirroredObject* mirrored_object = nullptr;
  char* dptr = nullptr;
  size_t size = 0;
  {
    const auto& stream = instr->mut_instr_chain()->stream();
    auto parallel_num = stream.thread().stream_rt_desc().stream_desc().parallel_num();
    FlatMsgView<CudaMallocHostInstruction> view;
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
  CudaCheck(cudaMallocHost(&dptr, size));
  mirrored_object->mutable_host_mem_buffer()->__Init__(size, dptr);
}
REGISTER_HOST_INSTRUCTION(kCudaMallocHostOpcode, CudaMallocHost);
COMMAND(RegisterInstrTypeId<HostStreamType>("CudaMallocHost", kCudaMallocHostOpcode, kRemote));

// clang-format off
FLAT_MSG_VIEW_BEGIN(MallocInstruction);
  FLAT_MSG_VIEW_DEFINE_PATTERN(MutableMirroredObjectOperand, mirrored_object_operand);
  FLAT_MSG_VIEW_DEFINE_PATTERN(uint64_t, size);
FLAT_MSG_VIEW_END(MallocInstruction);
// clang-format on

void Malloc(Instruction* instr) {
  MirroredObject* mirrored_object = nullptr;
  char* dptr = nullptr;
  size_t size = 0;
  {
    const auto& stream = instr->mut_instr_chain()->stream();
    auto parallel_num = stream.thread().stream_rt_desc().stream_desc().parallel_num();
    FlatMsgView<MallocInstruction> view;
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
  dptr = reinterpret_cast<char*>(std::malloc(size));
  mirrored_object->mutable_host_mem_buffer()->__Init__(size, dptr);
}
REGISTER_HOST_INSTRUCTION(kMallocOpcode, Malloc);
COMMAND(RegisterInstrTypeId<HostStreamType>("Malloc", kMallocOpcode, kRemote));

// clang-format off
FLAT_MSG_VIEW_BEGIN(CudaFreeHostInstruction);
  FLAT_MSG_VIEW_DEFINE_PATTERN(MutableMirroredObjectOperand, mirrored_object_operand);
FLAT_MSG_VIEW_END(CudaFreeHostInstruction);
// clang-format on

void CudaFreeHost(Instruction* instr) {
  MirroredObject* mirrored_object = nullptr;
  {
    const auto& stream = instr->mut_instr_chain()->stream();
    auto parallel_num = stream.thread().stream_rt_desc().stream_desc().parallel_num();
    FlatMsgView<CudaFreeHostInstruction> view;
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
  CudaCheck(cudaFreeHost(mirrored_object->mut_host_mem_buffer()->mut_data()));
  mirrored_object->clear_host_mem_buffer();
}
REGISTER_HOST_INSTRUCTION(kCudaFreeHostOpcode, CudaFreeHost);
COMMAND(RegisterInstrTypeId<HostStreamType>("CudaFreeHost", kCudaFreeHostOpcode, kRemote));

// clang-format off
FLAT_MSG_VIEW_BEGIN(FreeInstruction);
  FLAT_MSG_VIEW_DEFINE_PATTERN(MutableMirroredObjectOperand, mirrored_object_operand);
FLAT_MSG_VIEW_END(FreeInstruction);
// clang-format on

void Free(Instruction* instr) {
  MirroredObject* mirrored_object = nullptr;
  {
    const auto& stream = instr->mut_instr_chain()->stream();
    auto parallel_num = stream.thread().stream_rt_desc().stream_desc().parallel_num();
    FlatMsgView<FreeInstruction> view;
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
  std::free(mirrored_object->mut_host_mem_buffer()->mut_data());
  mirrored_object->clear_host_mem_buffer();
}
REGISTER_HOST_INSTRUCTION(kFreeOpcode, Free);
COMMAND(RegisterInstrTypeId<HostStreamType>("Free", kFreeOpcode, kRemote));

}  // namespace

const StreamTypeId HostStreamType::kStreamTypeId;

void HostStreamType::InitInstructionStatus(const Stream& stream,
                                           InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(NaiveInstrStatusQuerier) < kInstructionStatusBufferBytes, "");
  NaiveInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data());
}

void HostStreamType::DeleteInstructionStatus(const Stream& stream,
                                             InstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool HostStreamType::QueryInstructionStatusDone(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return NaiveInstrStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

ObjectMsgPtr<InstructionMsg> HostStreamType::CudaMallocHost(uint64_t logical_object_id,
                                                            size_t size) const {
  auto instr_msg = ObjectMsgPtr<InstructionMsg>::New();
  auto* instr_type_id = instr_msg->mutable_instr_type_id();
  instr_type_id->set_stream_type_id(kStreamTypeId);
  instr_type_id->set_opcode(HostInstrOpCode::kCudaMallocHostOpcode);
  {
    FlatMsgView<CudaMallocHostInstruction> view(instr_msg->mutable_operand());
    view->mutable_mirrored_object_operand()->mutable_operand()->__Init__(logical_object_id);
    view->set_size(size);
  }
  return instr_msg;
}

ObjectMsgPtr<InstructionMsg> HostStreamType::Malloc(uint64_t logical_object_id, size_t size) const {
  auto instr_msg = ObjectMsgPtr<InstructionMsg>::New();
  auto* instr_type_id = instr_msg->mutable_instr_type_id();
  instr_type_id->set_stream_type_id(kStreamTypeId);
  instr_type_id->set_opcode(HostInstrOpCode::kMallocOpcode);
  {
    FlatMsgView<CudaMallocHostInstruction> view(instr_msg->mutable_operand());
    view->mutable_mirrored_object_operand()->mutable_operand()->__Init__(logical_object_id);
    view->set_size(size);
  }
  return instr_msg;
}

ObjectMsgPtr<InstructionMsg> HostStreamType::CudaFreeHost(uint64_t logical_object_id) const {
  auto instr_msg = ObjectMsgPtr<InstructionMsg>::New();
  auto* instr_type_id = instr_msg->mutable_instr_type_id();
  instr_type_id->set_stream_type_id(kStreamTypeId);
  instr_type_id->set_opcode(HostInstrOpCode::kCudaFreeHostOpcode);
  {
    FlatMsgView<CudaFreeHostInstruction> view(instr_msg->mutable_operand());
    view->mutable_mirrored_object_operand()->mutable_operand()->__Init__(logical_object_id);
  }
  return instr_msg;
}

ObjectMsgPtr<InstructionMsg> HostStreamType::Free(uint64_t logical_object_id) const {
  auto instr_msg = ObjectMsgPtr<InstructionMsg>::New();
  auto* instr_type_id = instr_msg->mutable_instr_type_id();
  instr_type_id->set_stream_type_id(kStreamTypeId);
  instr_type_id->set_opcode(HostInstrOpCode::kFreeOpcode);
  {
    FlatMsgView<CudaFreeHostInstruction> view(instr_msg->mutable_operand());
    view->mutable_mirrored_object_operand()->mutable_operand()->__Init__(logical_object_id);
  }
  return instr_msg;
}

void HostStreamType::Run(InstrChain* instr_chain) const {
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(instr_chain->mut_instruction_list(), instruction) {
    auto opcode = instruction->mut_instr_msg()->instr_type_id().opcode();
    host_instr_table.at(opcode)(instruction);
  }
  auto* status_buffer = instr_chain->mut_status_buffer();
  NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data())->set_done();
}

COMMAND(RegisterStreamType<HostStreamType>());

}  // namespace vm
}  // namespace oneflow
