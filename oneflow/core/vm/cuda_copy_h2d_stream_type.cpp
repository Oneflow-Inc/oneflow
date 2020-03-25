#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/vm/cuda_instruction_status_querier.h"
#include "oneflow/core/vm/cuda_stream_handle_device_context.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {
namespace vm {

class CudaCopyH2DStreamType final : public StreamType {
 public:
  CudaCopyH2DStreamType() = default;
  ~CudaCopyH2DStreamType() = default;

  void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const override;

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

class CudaCopyH2DInstructionType final : public InstructionType {
 public:
  CudaCopyH2DInstructionType() = default;
  ~CudaCopyH2DInstructionType() override = default;

  using stream_type = CudaCopyH2DStreamType;

  // clang-format off
  FLAT_MSG_VIEW_BEGIN(CudaCopyH2DInstruction);
    FLAT_MSG_VIEW_DEFINE_PATTERN(MutableMirroredObjectOperand, dst);
    FLAT_MSG_VIEW_DEFINE_PATTERN(ConstMirroredObjectOperand, src);
    FLAT_MSG_VIEW_DEFINE_PATTERN(uint64_t, size);
  FLAT_MSG_VIEW_END(CudaCopyH2DInstruction);
  // clang-format on

  void Compute(Instruction* instr) const override {
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
};
COMMAND(RegisterInstructionType<CudaCopyH2DInstructionType>("CopyH2D"));
COMMAND(RegisterInstructionType<CudaCopyH2DInstructionType>("CudaCopyH2D"));

}  // namespace

void CudaCopyH2DStreamType::InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx,
                                          Stream* stream) const {
  device_ctx->reset(new CudaStreamHandleDeviceCtx(stream->mut_callback_list()));
}

void CudaCopyH2DStreamType::InitInstructionStatus(const Stream& stream,
                                                  InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(CudaInstrStatusQuerier) < kInstructionStatusBufferBytes, "");
  CudaInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data(),
                                       stream.thread_ctx().device_id());
}

void CudaCopyH2DStreamType::DeleteInstructionStatus(const Stream& stream,
                                                    InstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool CudaCopyH2DStreamType::QueryInstructionStatusDone(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return CudaInstrStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void CudaCopyH2DStreamType::Compute(InstrChain* instr_chain) const {
  auto* stream = instr_chain->mut_stream();
  cudaSetDevice(stream->thread_ctx().device_id());
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(instr_chain->mut_instruction_list(), instruction) {
    const auto& instr_type_id = instruction->mut_instr_msg()->instr_type_id();
    CHECK_EQ(instr_type_id.stream_type_id().interpret_type(), InterpretType::kCompute);
    LookupInstructionType(instr_type_id)->Compute(instruction);
  }
  stream->mut_callback_list()->MoveTo(instr_chain->mut_callback_list());
  char* data_ptr = instr_chain->mut_status_buffer()->mut_buffer()->mut_data();
  CudaInstrStatusQuerier::MutCast(data_ptr)->SetLaunched(stream->device_ctx().get());
}

ObjectMsgPtr<StreamDesc> CudaCopyH2DStreamType::MakeRemoteStreamDesc(
    const Resource& resource, int64_t this_machine_id) const {
  std::size_t device_num = resource.gpu_device_num();
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->mutable_stream_type_id()->__Init__(typeid(CudaCopyH2DStreamType));
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(device_num);
  ret->set_num_streams_per_thread(1);
  ret->set_start_parallel_id(this_machine_id * device_num);
  return ret;
}

}  // namespace vm
}  // namespace oneflow
