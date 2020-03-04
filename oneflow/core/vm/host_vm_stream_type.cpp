#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/host_vm_stream_type.h"
#include "oneflow/core/vm/vm_instruction.msg.h"
#include "oneflow/core/vm/naive_vm_instruction_status_querier.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace {

static const VmStreamTypeId kHostVmStreamTypeId = 1;

enum HostInstrOpCode { kCudaHostMallocOpcode = 0 };

typedef void (*HostInstrFunc)(VmInstrChain*, VmInstructionMsg*);
std::vector<HostInstrFunc> host_instr_table;

#define REGISTER_HOST_INSTRUCTION(op_code, function_name) \
  COMMAND({                                               \
    host_instr_table.resize(op_code + 1);                 \
    host_instr_table.at(op_code) = &function_name;        \
  })

// clang-format off
FLAT_MSG_VIEW_BEGIN(CudaHostMallocInstruction);
FLAT_MSG_VIEW_DEFINE_PATTERN(MutableLogicalObjectId, symbol);
FLAT_MSG_VIEW_DEFINE_PATTERN(uint64_t, size);
FLAT_MSG_VIEW_DEFINE_PATTERN(uint64_t, device_id);
FLAT_MSG_VIEW_END(CudaHostMallocInstruction);
// clang-format on

void VmCudaHostMalloc(VmInstrChain* vm_instr_chain, VmInstructionMsg* vm_instr_msg) {
  FlatMsgView<CudaHostMallocInstruction> view;
  CHECK(view->Match(vm_instr_msg->mut_vm_instruction_proto()->mut_operand()));
  char* dptr = nullptr;
  cudaSetDevice(view->device_id());
  CudaCheck(cudaMallocHost(&dptr, view->size()));
  TODO();
}
REGISTER_HOST_INSTRUCTION(kCudaHostMallocOpcode, VmCudaHostMalloc);

}  // namespace

void HostVmStreamType::InitVmInstructionStatus(const VmStream& vm_stream,
                                               VmInstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(NaiveVmInstrStatusQuerier) < kVmInstructionStatusBufferLength, "");
  NaiveVmInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data());
}

void HostVmStreamType::DeleteVmInstructionStatus(const VmStream& vm_stream,
                                                 VmInstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool HostVmStreamType::QueryVmInstructionStatusDone(
    const VmStream& vm_stream, const VmInstructionStatusBuffer& status_buffer) const {
  return NaiveVmInstrStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void HostVmStreamType::Run(VmStream* vm_stream, VmInstrChainPackage* vm_instr_chain_pkg) const {
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(vm_instr_chain_pkg->mut_vm_instr_chain_list(), chain) {
    OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(chain->mut_vm_instruction_list(), vm_instruction) {
      auto* vm_instruction_msg = vm_instruction->mut_vm_instruction_msg();
      auto opcode = vm_instruction_msg->vm_instruction_proto().opcode();
      host_instr_table.at(opcode)(chain, vm_instruction_msg);
    }
  }
  auto* status_buffer = vm_instr_chain_pkg->mut_status_buffer();
  NaiveVmInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data())->set_done();
}

}  // namespace oneflow
