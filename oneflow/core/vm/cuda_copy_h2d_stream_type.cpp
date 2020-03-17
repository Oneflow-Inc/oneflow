#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/cuda_copy_h2d_stream_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/thread.msg.h"
#include "oneflow/core/vm/cuda_instruction_status_querier.h"
#include "oneflow/core/vm/cuda_stream_handle_device_context.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {
namespace vm {

namespace {

// clang-format off
FLAT_MSG_VIEW_BEGIN(CudaCopyH2DInstruction);
  FLAT_MSG_VIEW_DEFINE_PATTERN(MutableMirroredObjectOperand, dst);
  FLAT_MSG_VIEW_DEFINE_PATTERN(ConstMirroredObjectOperand, src);
  FLAT_MSG_VIEW_DEFINE_PATTERN(uint64_t, size);
FLAT_MSG_VIEW_END(CudaCopyH2DInstruction);
// clang-format on

void CudaCopyH2D(Instruction* instr) {
  void* dst = nullptr;
  const void* src = nullptr;
  size_t size = 0;
  const auto& stream = instr->mut_instr_chain()->stream();
  {
    FlatMsgView<CudaCopyH2DInstruction> view;
    CHECK(view->Match(instr->mut_instr_msg()->mut_operand()));
    size = view->size();
    auto* dst_mirrored_obj =
        instr->FindMirroredObjectByOperand(view->dst().operand(), stream.parallel_id());
    CHECK_NOTNULL(dst_mirrored_obj);
    dst = dst_mirrored_obj->mut_cuda_mem_buffer()->mut_data();
    auto* src_mirrored_obj =
        instr->FindMirroredObjectByOperand(view->src().operand(), stream.parallel_id());
    CHECK_NOTNULL(src_mirrored_obj);
    src = src_mirrored_obj->mut_host_mem_buffer()->mut_data();
  }
  Memcpy<DeviceType::kGPU>(stream.device_ctx().get(), dst, src, size,
                           cudaMemcpyKind::cudaMemcpyHostToDevice);
}

}  // namespace

const StreamTypeId CudaCopyH2DStreamType::kStreamTypeId;

ObjectMsgPtr<InstructionMsg> CudaCopyH2DStreamType::Copy(uint64_t dst, uint64_t src,
                                                         size_t size) const {
  auto instr_msg = ObjectMsgPtr<InstructionMsg>::New();
  auto* instr_type_id = instr_msg->mutable_instr_type_id();
  instr_type_id->set_stream_type_id(kStreamTypeId);
  instr_type_id->set_opcode(0);
  {
    FlatMsgView<CudaCopyH2DInstruction> view(instr_msg->mutable_operand());
    view->mutable_dst()->mutable_operand()->__Init__(dst);
    view->mutable_src()->mutable_operand()->__Init__(src);
    view->set_size(size);
  }
  return instr_msg;
}

void CudaCopyH2DStreamType::InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx,
                                          Stream* stream) const {
  device_ctx->reset(new CudaStreamHandleDeviceCtx(stream->mut_callback_list()));
}

void CudaCopyH2DStreamType::InitInstructionStatus(const Stream& stream,
                                                  InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(CudaInstrStatusQuerier) < kInstructionStatusBufferBytes, "");
  CudaInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data(),
                                       stream.thread().device_id());
}

void CudaCopyH2DStreamType::DeleteInstructionStatus(const Stream& stream,
                                                    InstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool CudaCopyH2DStreamType::QueryInstructionStatusDone(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return CudaInstrStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void CudaCopyH2DStreamType::Run(InstrChain* instr_chain) const {
  auto* stream = instr_chain->mut_stream();
  cudaSetDevice(stream->thread().device_id());
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(instr_chain->mut_instruction_list(), instruction) {
    CudaCopyH2D(instruction);
  }
  stream->mut_callback_list()->MoveTo(instr_chain->mut_callback_list());
  char* data_ptr = instr_chain->mut_status_buffer()->mut_buffer()->mut_data();
  CudaInstrStatusQuerier::MutCast(data_ptr)->SetLaunched(stream->device_ctx().get());
}

COMMAND(RegisterStreamType<CudaCopyH2DStreamType>());
COMMAND(RegisterInstrTypeId<CudaCopyH2DStreamType>("CopyH2D", 0, kRemote));
COMMAND(RegisterInstrTypeId<CudaCopyH2DStreamType>("CudaCopyH2D", 0, kRemote));

}  // namespace vm
}  // namespace oneflow
