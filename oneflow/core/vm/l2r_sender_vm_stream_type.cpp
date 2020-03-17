#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/l2r_sender_vm_stream_type.h"
#include "oneflow/core/vm/vm_instruction.msg.h"
#include "oneflow/core/vm/vm_stream.msg.h"
#include "oneflow/core/vm/localhost_transporter.h"
#include "oneflow/core/vm/transporter_device_context.h"

namespace oneflow {
namespace vm {

namespace {

Transporter* GetTransporter(InstrChain* vm_instr_chain) {
  auto* device_ctx =
      dynamic_cast<TransporterDeviceCtx*>(vm_instr_chain->mut_vm_stream()->device_ctx().get());
  CHECK_NOTNULL(device_ctx);
  return device_ctx->mut_transporter();
}

// clang-format off
FLAT_MSG_VIEW_BEGIN(L2RSenderInstruction);
  FLAT_MSG_VIEW_DEFINE_PATTERN(ConstMirroredObjectOperand, src);
  FLAT_MSG_VIEW_DEFINE_PATTERN(uint64_t, logical_token);
  FLAT_MSG_VIEW_DEFINE_PATTERN(uint64_t, size);
FLAT_MSG_VIEW_END(L2RSenderInstruction);
// clang-format on

void MakeSendRequests(Instruction* vm_instr, TransportKey2SendRequest* transport_key2send_request) {
  auto* vm_instr_chain = vm_instr->mut_vm_instr_chain();
  FlatMsg<TransportDataToken> data_token;
  const char* data_ptr = nullptr;
  size_t data_size = 0;
  {
    const auto& vm_stream = vm_instr_chain->vm_stream();
    FlatMsgView<L2RSenderInstruction> view;
    CHECK(view->Match(vm_instr->mut_vm_instr_msg()->mut_operand()));
    data_token->mutable_mirrored_token()->set_logical_token(view->logical_token());
    data_token->mutable_mirrored_token()->set_parallel_id(vm_stream.parallel_id());
    data_size = view->size();
    auto* src_mirrored_obj =
        vm_instr->FindMirroredObjectByOperand(view->src().operand(), vm_stream.parallel_id());
    CHECK_NOTNULL(src_mirrored_obj);
    CHECK(src_mirrored_obj->has_host_mem_buffer());
    data_ptr = &src_mirrored_obj->host_mem_buffer().data();
  }
  std::atomic<int64_t>* incomplete_cnt = nullptr;
  {
    char* buffer_ptr = vm_instr_chain->mut_status_buffer()->mut_buffer()->mut_data();
    incomplete_cnt = reinterpret_cast<std::atomic<int64_t>*>(buffer_ptr);
  }
  GetTransporter(vm_instr->mut_vm_instr_chain())
      ->MakeSendTransportRequest(data_token.Get(), data_ptr, data_size, incomplete_cnt,
                                 transport_key2send_request);
}

}  // namespace

const StreamTypeId L2RSenderStreamType::kStreamTypeId;

ObjectMsgPtr<InstructionMsg> L2RSenderStreamType::Send(uint64_t logical_token, uint64_t src,
                                                       size_t size) const {
  auto vm_instr_msg = ObjectMsgPtr<InstructionMsg>::New();
  auto* vm_instr_id = vm_instr_msg->mutable_vm_instr_id();
  vm_instr_id->set_vm_stream_type_id(kStreamTypeId);
  vm_instr_id->set_opcode(0);
  {
    FlatMsgView<L2RSenderInstruction> view(vm_instr_msg->mutable_operand());
    view->mutable_src()->mutable_operand()->__Init__(src);
    view->set_logical_token(logical_token);
    view->set_size(size);
  }
  return vm_instr_msg;
}

void L2RSenderStreamType::InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx,
                                        Stream* vm_stream) const {
  if (vm_stream->machine_id() != 0) { TODO(); }
  device_ctx->reset(new TransporterDeviceCtx(new LocalhostTransporter()));
}

static const int64_t kRefCntInitVal = 1;

void L2RSenderStreamType::InitInstructionStatus(const Stream& vm_stream,
                                                InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(std::atomic<int64_t>) < kInstructionStatusBufferBytes, "");
  new (status_buffer->mut_buffer()->mut_data()) std::atomic<int64_t>(kRefCntInitVal);
}

void L2RSenderStreamType::DeleteInstructionStatus(const Stream& vm_stream,
                                                  InstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool L2RSenderStreamType::QueryInstructionStatusDone(
    const Stream& vm_stream, const InstructionStatusBuffer& status_buffer) const {
  return *reinterpret_cast<const std::atomic<int64_t>*>(status_buffer.buffer().data()) == 0;
}

void L2RSenderStreamType::Run(InstrChain* vm_instr_chain) const {
  char* data_ptr = vm_instr_chain->mut_status_buffer()->mut_buffer()->mut_data();
  TransportKey2SendRequest transport_key2send_request;
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(vm_instr_chain->mut_vm_instruction_list(), vm_instruction) {
    MakeSendRequests(vm_instruction, &transport_key2send_request);
  }
  // inital val is one
  *reinterpret_cast<std::atomic<int64_t>*>(data_ptr) +=
      transport_key2send_request.size() - kRefCntInitVal;
  GetTransporter(vm_instr_chain)->Transport(&transport_key2send_request);
}

COMMAND(RegisterStreamType<L2RSenderStreamType>());
COMMAND(RegisterInstructionId<L2RSenderStreamType>("L2RSend", 0, kRemote));
COMMAND(RegisterInstructionId<L2RSenderStreamType>("L2RLocalSend", 0, kLocal));

}  // namespace vm
}  // namespace oneflow
