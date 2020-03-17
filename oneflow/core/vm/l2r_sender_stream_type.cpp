#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/l2r_sender_stream_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/localhost_transporter.h"
#include "oneflow/core/vm/transporter_device_context.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {
namespace vm {

namespace {

Transporter* GetTransporter(InstrChain* instr_chain) {
  auto* device_ctx =
      dynamic_cast<TransporterDeviceCtx*>(instr_chain->mut_stream()->device_ctx().get());
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

void MakeSendRequests(Instruction* instr, TransportKey2SendRequest* transport_key2send_request) {
  auto* instr_chain = instr->mut_instr_chain();
  FlatMsg<TransportDataToken> data_token;
  const char* data_ptr = nullptr;
  size_t data_size = 0;
  {
    const auto& stream = instr_chain->stream();
    FlatMsgView<L2RSenderInstruction> view;
    CHECK(view->Match(instr->mut_instr_msg()->mut_operand()));
    data_token->mutable_mirrored_token()->set_logical_token(view->logical_token());
    data_token->mutable_mirrored_token()->set_parallel_id(stream.parallel_id());
    data_size = view->size();
    auto* src_mirrored_obj =
        instr->FindMirroredObjectByOperand(view->src().operand(), stream.parallel_id());
    CHECK_NOTNULL(src_mirrored_obj);
    CHECK(src_mirrored_obj->has_host_mem_buffer());
    data_ptr = &src_mirrored_obj->host_mem_buffer().data();
  }
  std::atomic<int64_t>* incomplete_cnt = nullptr;
  {
    char* buffer_ptr = instr_chain->mut_status_buffer()->mut_buffer()->mut_data();
    incomplete_cnt = reinterpret_cast<std::atomic<int64_t>*>(buffer_ptr);
  }
  GetTransporter(instr->mut_instr_chain())
      ->MakeSendTransportRequest(data_token.Get(), data_ptr, data_size, incomplete_cnt,
                                 transport_key2send_request);
}

}  // namespace

const StreamTypeId L2RSenderStreamType::kStreamTypeId;

ObjectMsgPtr<InstructionMsg> L2RSenderStreamType::Send(uint64_t logical_token, uint64_t src,
                                                       size_t size) const {
  auto instr_msg = ObjectMsgPtr<InstructionMsg>::New();
  auto* instr_type_id = instr_msg->mutable_instr_type_id();
  instr_type_id->set_stream_type_id(kStreamTypeId);
  instr_type_id->set_opcode(0);
  {
    FlatMsgView<L2RSenderInstruction> view(instr_msg->mutable_operand());
    view->mutable_src()->mutable_operand()->__Init__(src);
    view->set_logical_token(logical_token);
    view->set_size(size);
  }
  return instr_msg;
}

void L2RSenderStreamType::InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx,
                                        Stream* stream) const {
  if (stream->machine_id() != 0) { TODO(); }
  device_ctx->reset(new TransporterDeviceCtx(new LocalhostTransporter()));
}

static const int64_t kRefCntInitVal = 1;

void L2RSenderStreamType::InitInstructionStatus(const Stream& stream,
                                                InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(std::atomic<int64_t>) < kInstructionStatusBufferBytes, "");
  new (status_buffer->mut_buffer()->mut_data()) std::atomic<int64_t>(kRefCntInitVal);
}

void L2RSenderStreamType::DeleteInstructionStatus(const Stream& stream,
                                                  InstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool L2RSenderStreamType::QueryInstructionStatusDone(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return *reinterpret_cast<const std::atomic<int64_t>*>(status_buffer.buffer().data()) == 0;
}

void L2RSenderStreamType::Run(InstrChain* instr_chain) const {
  char* data_ptr = instr_chain->mut_status_buffer()->mut_buffer()->mut_data();
  TransportKey2SendRequest transport_key2send_request;
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(instr_chain->mut_instruction_list(), instruction) {
    MakeSendRequests(instruction, &transport_key2send_request);
  }
  // inital val is one
  *reinterpret_cast<std::atomic<int64_t>*>(data_ptr) +=
      transport_key2send_request.size() - kRefCntInitVal;
  GetTransporter(instr_chain)->Transport(&transport_key2send_request);
}

ObjectMsgPtr<StreamDesc> L2RSenderStreamType::MakeRemoteStreamDesc(const Resource& resource,
                                                                   int64_t this_machine_id) const {
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->set_stream_type_id(kStreamTypeId);
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(1);
  ret->set_num_streams_per_thread(1);
  ret->set_start_parallel_id(this_machine_id);
  return ret;
}

ObjectMsgPtr<StreamDesc> L2RSenderStreamType::MakeLocalStreamDesc(const Resource& resource) const {
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->set_stream_type_id(kStreamTypeId);
  ret->set_num_machines(resource.machine_num());
  ret->set_num_streams_per_machine(1);
  ret->set_num_streams_per_thread(1);
  ret->set_start_parallel_id(0);
  return ret;
}

COMMAND(RegisterStreamType<L2RSenderStreamType>());
COMMAND(RegisterInstrTypeId<L2RSenderStreamType>("L2RSend", 0, kRemote));
COMMAND(RegisterInstrTypeId<L2RSenderStreamType>("L2RLocalSend", 0, kLocal));

}  // namespace vm
}  // namespace oneflow
