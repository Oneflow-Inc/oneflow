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

class L2RSenderStreamType final : public StreamType {
 public:
  L2RSenderStreamType() = default;
  ~L2RSenderStreamType() = default;

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

class L2RSenderInstructionType final : public InstructionType {
 public:
  L2RSenderInstructionType() = default;
  ~L2RSenderInstructionType() override = default;

  using stream_type = L2RSenderStreamType;

  void Infer(Instruction* instruction) const override { /* do nothing */
  }
  void Compute(Instruction* instruction) const override { UNIMPLEMENTED(); }
};
COMMAND(RegisterInstructionType<L2RSenderInstructionType>("L2RSend"));

Transporter* GetTransporter(Instruction* instruction) {
  auto* device_ctx =
      dynamic_cast<TransporterDeviceCtx*>(instruction->mut_stream()->device_ctx().get());
  CHECK_NOTNULL(device_ctx);
  return device_ctx->mut_transporter();
}

// clang-format off
FLAT_MSG_VIEW_BEGIN(L2RSenderInstruction);
  FLAT_MSG_VIEW_DEFINE_PATTERN(ConstOperand, src);
  FLAT_MSG_VIEW_DEFINE_PATTERN(int64_t, logical_token);
  FLAT_MSG_VIEW_DEFINE_PATTERN(int64_t, size);
FLAT_MSG_VIEW_END(L2RSenderInstruction);
// clang-format on

void MakeSendRequests(Instruction* instruction,
                      TransportKey2SendRequest* transport_key2send_request) {
  FlatMsg<TransportDataToken> data_token;
  const char* data_ptr = nullptr;
  size_t data_size = 0;
  {
    const auto& stream = instruction->stream();
    FlatMsgView<L2RSenderInstruction> view;
    CHECK(view.Match(instruction->instr_msg().operand()));
    data_token->mutable_mirrored_token()->set_logical_token(view->logical_token());
    data_token->mutable_mirrored_token()->set_global_device_id(stream.global_device_id());
    data_size = view->size();
    const auto& src_buffer_type =
        instruction->operand_type(view->src())->Get<MemBufferObjectType>();
    CHECK_LE(data_size, src_buffer_type.size());
    CHECK(src_buffer_type.mem_case().has_host_mem());
    const auto& src_buffer_value =
        instruction->operand_value(view->src())->Get<MemBufferObjectValue>();
    data_ptr = src_buffer_value.data();
  }
  std::atomic<int64_t>* incomplete_cnt = nullptr;
  {
    char* buffer_ptr = instruction->mut_status_buffer()->mut_buffer()->mut_data();
    incomplete_cnt = reinterpret_cast<std::atomic<int64_t>*>(buffer_ptr);
  }
  GetTransporter(instruction)
      ->MakeSendTransportRequest(data_token.Get(), data_ptr, data_size, incomplete_cnt,
                                 transport_key2send_request);
}

}  // namespace

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

void L2RSenderStreamType::Compute(Instruction* instruction) const {
  char* data_ptr = instruction->mut_status_buffer()->mut_buffer()->mut_data();
  TransportKey2SendRequest transport_key2send_request;
  MakeSendRequests(instruction, &transport_key2send_request);
  // inital val is one
  *reinterpret_cast<std::atomic<int64_t>*>(data_ptr) +=
      transport_key2send_request.size() - kRefCntInitVal;
  GetTransporter(instruction)->Transport(&transport_key2send_request);
}

ObjectMsgPtr<StreamDesc> L2RSenderStreamType::MakeStreamDesc(const Resource& resource,
                                                             int64_t this_machine_id) const {
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->mutable_stream_type_id()->__Init__(LookupStreamType4TypeIndex<L2RSenderStreamType>());
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(1);
  ret->set_num_streams_per_thread(1);
  ret->set_start_global_device_id(this_machine_id);
  return ret;
}

}  // namespace vm
}  // namespace oneflow
