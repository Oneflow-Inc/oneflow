#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/localhost_transporter.h"
#include "oneflow/core/vm/transporter_device_context.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {
namespace vm {

class L2RSenderStreamType final : public StreamType {
 public:
  L2RSenderStreamType() = default;
  ~L2RSenderStreamType() = default;

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
  ObjectMsgPtr<StreamDesc> MakeLocalStreamDesc(const Resource& resource) const override;
};

namespace {

class L2RSenderInstructionType final : public InstructionType {
 public:
  L2RSenderInstructionType() = default;
  ~L2RSenderInstructionType() override = default;

  using stream_type = L2RSenderStreamType;

  void Infer(InstrCtx* instr_ctx) const override { /* do nothing */
  }
  void Compute(InstrCtx* instr_ctx) const override { UNIMPLEMENTED(); }
};
COMMAND(RegisterInstructionType<L2RSenderInstructionType>("L2RSend"));
COMMAND(RegisterLocalInstructionType<L2RSenderInstructionType>("L2RLocalSend"));

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

void MakeSendRequests(InstrCtx* instr_ctx, TransportKey2SendRequest* transport_key2send_request) {
  auto* instr_chain = instr_ctx->mut_instr_chain();
  FlatMsg<TransportDataToken> data_token;
  const char* data_ptr = nullptr;
  size_t data_size = 0;
  {
    const auto& stream = instr_chain->stream();
    FlatMsgView<L2RSenderInstruction> view;
    CHECK(view->Match(instr_ctx->mut_instr_msg()->mut_operand()));
    data_token->mutable_mirrored_token()->set_logical_token(view->logical_token());
    data_token->mutable_mirrored_token()->set_parallel_id(stream.parallel_id());
    data_size = view->size();
    auto* src_mirrored_obj =
        instr_ctx->FindMirroredObjectByOperand(view->src().operand(), stream.parallel_id());
    CHECK_NOTNULL(src_mirrored_obj);
    CHECK(src_mirrored_obj->has_host_mem_buffer());
    data_ptr = &src_mirrored_obj->host_mem_buffer().data();
  }
  std::atomic<int64_t>* incomplete_cnt = nullptr;
  {
    char* buffer_ptr = instr_chain->mut_status_buffer()->mut_buffer()->mut_data();
    incomplete_cnt = reinterpret_cast<std::atomic<int64_t>*>(buffer_ptr);
  }
  GetTransporter(instr_ctx->mut_instr_chain())
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

void L2RSenderStreamType::Compute(InstrChain* instr_chain) const {
  char* data_ptr = instr_chain->mut_status_buffer()->mut_buffer()->mut_data();
  TransportKey2SendRequest transport_key2send_request;
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(instr_chain->mut_instr_ctx_list(), instr_ctx) {
    MakeSendRequests(instr_ctx, &transport_key2send_request);
  }
  // inital val is one
  *reinterpret_cast<std::atomic<int64_t>*>(data_ptr) +=
      transport_key2send_request.size() - kRefCntInitVal;
  GetTransporter(instr_chain)->Transport(&transport_key2send_request);
}

ObjectMsgPtr<StreamDesc> L2RSenderStreamType::MakeRemoteStreamDesc(const Resource& resource,
                                                                   int64_t this_machine_id) const {
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->mutable_stream_type_id()->__Init__(LookupStreamType4TypeIndex<L2RSenderStreamType>());
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(1);
  ret->set_num_streams_per_thread(1);
  ret->set_start_parallel_id(this_machine_id);
  return ret;
}

ObjectMsgPtr<StreamDesc> L2RSenderStreamType::MakeLocalStreamDesc(const Resource& resource) const {
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->mutable_stream_type_id()->__Init__(LookupStreamType4TypeIndex<L2RSenderStreamType>());
  ret->set_num_machines(resource.machine_num());
  ret->set_num_streams_per_machine(1);
  ret->set_num_streams_per_thread(1);
  ret->set_start_parallel_id(0);
  return ret;
}

}  // namespace vm
}  // namespace oneflow
