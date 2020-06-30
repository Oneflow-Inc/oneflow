#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/localhost_transporter.h"
#include "oneflow/core/vm/transporter_device_context.h"
#include "oneflow/core/vm/mem_buffer_object.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {
namespace vm {

class L2RReceiverStreamType final : public StreamType {
 public:
  L2RReceiverStreamType() = default;
  ~L2RReceiverStreamType() = default;

  const char* device_tag() const override { return "cpu"; }

  void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const override;

  void InitInstructionStatus(const Stream& stream,
                             InstructionStatusBuffer* status_buffer) const override;
  void DeleteInstructionStatus(const Stream& stream,
                               InstructionStatusBuffer* status_buffer) const override;
  bool QueryInstructionStatusDone(const Stream& stream,
                                  const InstructionStatusBuffer& status_buffer) const override;
  void Compute(Instruction* instruction) const override;
  ObjectMsgPtr<StreamDesc> MakeStreamDesc(const Resource& resource,
                                          int64_t this_machine_id) const override;
};

namespace {

class L2RReceiverInstructionType final : public InstructionType {
 public:
  L2RReceiverInstructionType() = default;
  ~L2RReceiverInstructionType() override = default;

  using stream_type = L2RReceiverStreamType;

  void Infer(Instruction* instruction) const override { /* do nothing */
  }
  void Compute(Instruction* instruction) const override { UNIMPLEMENTED(); }
};
COMMAND(RegisterInstructionType<L2RReceiverInstructionType>("L2RReceive"));

Transporter* GetTransporter(Instruction* instruction) {
  auto* device_ctx =
      dynamic_cast<TransporterDeviceCtx*>(instruction->mut_stream()->device_ctx().get());
  CHECK_NOTNULL(device_ctx);
  return device_ctx->mut_transporter();
}

// clang-format off
FLAT_MSG_VIEW_BEGIN(L2RReceiverInstruction);
  FLAT_MSG_VIEW_DEFINE_PATTERN(MutOperand, dst);
  FLAT_MSG_VIEW_DEFINE_PATTERN(int64_t, logical_token);
  FLAT_MSG_VIEW_DEFINE_PATTERN(int64_t, size);
FLAT_MSG_VIEW_END(L2RReceiverInstruction);
// clang-format on

void MakeReceiveRequests(Instruction* instruction,
                         TransportKey2ReceiveRequest* transport_key2receive_request) {
  FlatMsg<TransportDataToken> data_token;
  char* data_ptr = nullptr;
  size_t data_size = 0;
  {
    const auto& stream = instruction->stream();
    FlatMsgView<L2RReceiverInstruction> view;
    CHECK(view.Match(instruction->instr_msg().operand()));
    data_token->mutable_mirrored_token()->set_logical_token(view->logical_token());
    data_token->mutable_mirrored_token()->set_global_device_id(stream.global_device_id());
    data_size = view->size();
    const auto& dst_buffer_type =
        instruction->operand_type(view->dst())->Get<MemBufferObjectType>();
    CHECK_LE(data_size, dst_buffer_type.size());
    CHECK(dst_buffer_type.mem_case().has_host_mem());
    auto* dst_buffer_value =
        instruction->mut_operand_value(view->dst())->Mut<MemBufferObjectValue>();
    data_ptr = dst_buffer_value->mut_data();
  }
  std::atomic<int64_t>* incomplete_cnt = nullptr;
  {
    char* buffer_ptr = instruction->mut_status_buffer()->mut_buffer()->mut_data();
    incomplete_cnt = reinterpret_cast<std::atomic<int64_t>*>(buffer_ptr);
  }
  GetTransporter(instruction)
      ->MakeReceiveTransportRequest(data_token.Get(), data_ptr, data_size, incomplete_cnt,
                                    transport_key2receive_request);
}

}  // namespace

void L2RReceiverStreamType::InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx,
                                          Stream* stream) const {
  if (stream->machine_id() != 0) { TODO(); }
  device_ctx->reset(new TransporterDeviceCtx(new LocalhostTransporter()));
}

static const int64_t kRefCntInitVal = 1;

void L2RReceiverStreamType::InitInstructionStatus(const Stream& stream,
                                                  InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(std::atomic<int64_t>) < kInstructionStatusBufferBytes, "");
  new (status_buffer->mut_buffer()->mut_data()) std::atomic<int64_t>(kRefCntInitVal);
}

void L2RReceiverStreamType::DeleteInstructionStatus(const Stream& stream,
                                                    InstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool L2RReceiverStreamType::QueryInstructionStatusDone(
    const Stream& stream, const InstructionStatusBuffer& status_buffer) const {
  return *reinterpret_cast<const std::atomic<int64_t>*>(status_buffer.buffer().data()) == 0;
}

void L2RReceiverStreamType::Compute(Instruction* instruction) const {
  char* data_ptr = instruction->mut_status_buffer()->mut_buffer()->mut_data();
  TransportKey2ReceiveRequest transport_key2receive_request;
  MakeReceiveRequests(instruction, &transport_key2receive_request);
  // inital val is one
  *reinterpret_cast<std::atomic<int64_t>*>(data_ptr) +=
      transport_key2receive_request.size() - kRefCntInitVal;
  GetTransporter(instruction)->Transport(&transport_key2receive_request);
}

ObjectMsgPtr<StreamDesc> L2RReceiverStreamType::MakeStreamDesc(const Resource& resource,
                                                               int64_t this_machine_id) const {
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->mutable_stream_type_id()->__Init__(LookupStreamType4TypeIndex<L2RReceiverStreamType>());
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(1);
  ret->set_num_streams_per_thread(1);
  ret->set_start_global_device_id(this_machine_id);
  return ret;
}

}  // namespace vm
}  // namespace oneflow
