#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/cuda_copy_h2d_vm_stream_type.h"
#include "oneflow/core/vm/vm_instruction.msg.h"
#include "oneflow/core/vm/vm_stream.msg.h"
#include "oneflow/core/vm/vm_thread.msg.h"
#include "oneflow/core/vm/cuda_vm_instruction_status_querier.h"
#include "oneflow/core/device/cuda_stream_handle_device_context.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

namespace {

// clang-format off
FLAT_MSG_VIEW_BEGIN(CudaCopyH2DInstruction);
  FLAT_MSG_VIEW_DEFINE_PATTERN(MutableMirroredObjectOperand, dst);
  FLAT_MSG_VIEW_DEFINE_PATTERN(ConstMirroredObjectOperand, src);
  FLAT_MSG_VIEW_DEFINE_PATTERN(uint64_t, size);
FLAT_MSG_VIEW_END(CudaCopyH2DInstruction);
// clang-format on

void VmCudaCopyH2D(VmInstruction* vm_instr) {
  void* dst = nullptr;
  const void* src = nullptr;
  size_t size = 0;
  const auto& vm_stream = vm_instr->mut_vm_instr_chain()->vm_stream();
  {
    FlatMsgView<CudaCopyH2DInstruction> view;
    CHECK(view->Match(vm_instr->mut_vm_instr_msg()->mut_vm_instruction_proto()->mut_operand()));
    size = view->size();
    auto* dst_mirrored_obj =
        vm_instr->FindMirroredObjectByOperand(view->dst().operand(), vm_stream.parallel_id());
    CHECK_NOTNULL(dst_mirrored_obj);
    dst = dst_mirrored_obj->mut_cuda_mem_buffer()->mut_data();
    auto* src_mirrored_obj =
        vm_instr->FindMirroredObjectByOperand(view->src().operand(), vm_stream.parallel_id());
    CHECK_NOTNULL(src_mirrored_obj);
    src = src_mirrored_obj->mut_host_mem_buffer()->mut_data();
  }
  Memcpy<DeviceType::kGPU>(vm_stream.device_ctx().get(), dst, src, size,
                           cudaMemcpyKind::cudaMemcpyHostToDevice);
}

}  // namespace

const VmStreamTypeId CudaCopyH2DVmStreamType::kVmStreamTypeId;

ObjectMsgPtr<VmInstructionMsg> CudaCopyH2DVmStreamType::Copy(uint64_t dst, uint64_t src,
                                                             size_t size) const {
  auto vm_instr_msg = ObjectMsgPtr<VmInstructionMsg>::New();
  auto* vm_instr_proto = vm_instr_msg->mutable_vm_instruction_proto();
  vm_instr_proto->set_vm_stream_type_id(kVmStreamTypeId);
  vm_instr_proto->set_opcode(0);
  vm_instr_proto->mutable_vm_stream_mask()->mutable_all_vm_stream_enabled();
  {
    FlatMsgView<CudaCopyH2DInstruction> view(vm_instr_proto->mutable_operand());
    view->mutable_dst()->mutable_operand()->__Init__(dst);
    view->mutable_src()->mutable_operand()->__Init__(src);
    view->set_size(size);
  }
  return vm_instr_msg;
}

void CudaCopyH2DVmStreamType::InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx,
                                            CallbackMsgListPtr callback_list) const {
  device_ctx->reset(new CudaStreamHandleDeviceCtx(callback_list));
}

void CudaCopyH2DVmStreamType::InitVmInstructionStatus(
    const VmStream& vm_stream, VmInstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(CudaVmInstrStatusQuerier) < kVmInstructionStatusBufferBytes, "");
  CudaVmInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data(),
                                         vm_stream.vm_thread().device_id());
}

void CudaCopyH2DVmStreamType::DeleteVmInstructionStatus(
    const VmStream& vm_stream, VmInstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool CudaCopyH2DVmStreamType::QueryVmInstructionStatusDone(
    const VmStream& vm_stream, const VmInstructionStatusBuffer& status_buffer) const {
  return CudaVmInstrStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void CudaCopyH2DVmStreamType::Run(VmInstrChain* vm_instr_chain) const {
  auto* vm_stream = vm_instr_chain->mut_vm_stream();
  cudaSetDevice(vm_stream->vm_thread().device_id());
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(vm_instr_chain->mut_vm_instruction_list(), vm_instruction) {
    VmCudaCopyH2D(vm_instruction);
  }
  vm_stream->mut_callback_list()->MoveTo(vm_instr_chain->mut_callback_list());
  char* data_ptr = vm_instr_chain->mut_status_buffer()->mut_buffer()->mut_data();
  CudaVmInstrStatusQuerier::MutCast(data_ptr)->SetLaunched(vm_stream->device_ctx().get());
}

COMMAND(RegisterVmStreamType<CudaCopyH2DVmStreamType>());

}  // namespace oneflow
