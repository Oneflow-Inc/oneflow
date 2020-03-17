#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/cuda_copy_d2h_vm_stream_type.h"
#include "oneflow/core/vm/vm_instruction.msg.h"
#include "oneflow/core/vm/vm_stream.msg.h"
#include "oneflow/core/vm/vm_thread.msg.h"
#include "oneflow/core/vm/cuda_vm_instruction_status_querier.h"
#include "oneflow/core/vm/cuda_stream_handle_device_context.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {
namespace vm {

namespace {

// clang-format off
FLAT_MSG_VIEW_BEGIN(CudaCopyD2HInstruction);
  FLAT_MSG_VIEW_DEFINE_PATTERN(MutableMirroredObjectOperand, dst);
  FLAT_MSG_VIEW_DEFINE_PATTERN(ConstMirroredObjectOperand, src);
  FLAT_MSG_VIEW_DEFINE_PATTERN(uint64_t, size);
FLAT_MSG_VIEW_END(CudaCopyD2HInstruction);
// clang-format on

void CudaCopyD2H(Instruction* vm_instr) {
  void* dst = nullptr;
  const void* src = nullptr;
  size_t size = 0;
  const auto& vm_stream = vm_instr->mut_vm_instr_chain()->vm_stream();
  {
    FlatMsgView<CudaCopyD2HInstruction> view;
    CHECK(view->Match(vm_instr->mut_vm_instr_msg()->mut_operand()));
    size = view->size();
    auto* dst_mirrored_obj =
        vm_instr->FindMirroredObjectByOperand(view->dst().operand(), vm_stream.parallel_id());
    CHECK_NOTNULL(dst_mirrored_obj);
    dst = dst_mirrored_obj->mut_host_mem_buffer()->mut_data();
    auto* src_mirrored_obj =
        vm_instr->FindMirroredObjectByOperand(view->src().operand(), vm_stream.parallel_id());
    CHECK_NOTNULL(src_mirrored_obj);
    src = src_mirrored_obj->mut_cuda_mem_buffer()->mut_data();
  }
  Memcpy<DeviceType::kGPU>(vm_stream.device_ctx().get(), dst, src, size,
                           cudaMemcpyKind::cudaMemcpyDeviceToHost);
}

}  // namespace

const StreamTypeId CudaCopyD2HStreamType::kStreamTypeId;

ObjectMsgPtr<InstructionMsg> CudaCopyD2HStreamType::Copy(uint64_t dst, uint64_t src,
                                                         size_t size) const {
  auto vm_instr_msg = ObjectMsgPtr<InstructionMsg>::New();
  auto* vm_instr_id = vm_instr_msg->mutable_vm_instr_id();
  vm_instr_id->set_vm_stream_type_id(kStreamTypeId);
  vm_instr_id->set_opcode(0);
  {
    FlatMsgView<CudaCopyD2HInstruction> view(vm_instr_msg->mutable_operand());
    view->mutable_dst()->mutable_operand()->__Init__(dst);
    view->mutable_src()->mutable_operand()->__Init__(src);
    view->set_size(size);
  }
  return vm_instr_msg;
}

void CudaCopyD2HStreamType::InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx,
                                          Stream* vm_stream) const {
  device_ctx->reset(new CudaStreamHandleDeviceCtx(vm_stream->mut_callback_list()));
}

void CudaCopyD2HStreamType::InitInstructionStatus(const Stream& vm_stream,
                                                  InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(CudaInstrStatusQuerier) < kInstructionStatusBufferBytes, "");
  CudaInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data(),
                                       vm_stream.vm_thread().device_id());
}

void CudaCopyD2HStreamType::DeleteInstructionStatus(const Stream& vm_stream,
                                                    InstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool CudaCopyD2HStreamType::QueryInstructionStatusDone(
    const Stream& vm_stream, const InstructionStatusBuffer& status_buffer) const {
  return CudaInstrStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void CudaCopyD2HStreamType::Run(InstrChain* vm_instr_chain) const {
  auto* vm_stream = vm_instr_chain->mut_vm_stream();
  cudaSetDevice(vm_stream->vm_thread().device_id());
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(vm_instr_chain->mut_vm_instruction_list(), vm_instruction) {
    CudaCopyD2H(vm_instruction);
  }
  vm_stream->mut_callback_list()->MoveTo(vm_instr_chain->mut_callback_list());
  char* data_ptr = vm_instr_chain->mut_status_buffer()->mut_buffer()->mut_data();
  CudaInstrStatusQuerier::MutCast(data_ptr)->SetLaunched(vm_stream->device_ctx().get());
}

COMMAND(RegisterStreamType<CudaCopyD2HStreamType>());
COMMAND(RegisterInstructionId<CudaCopyD2HStreamType>("CopyD2H", 0, kRemote));
COMMAND(RegisterInstructionId<CudaCopyD2HStreamType>("CudaCopyD2H", 0, kRemote));

}  // namespace vm
}  // namespace oneflow
