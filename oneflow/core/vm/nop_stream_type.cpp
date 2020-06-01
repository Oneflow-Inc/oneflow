#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/stream_type.h"
#include "oneflow/core/vm/instruction_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {
namespace vm {

class NopStreamType final : public StreamType {
 public:
  NopStreamType() = default;
  ~NopStreamType() = default;

  const char* device_tag() const override { return "cpu"; }

  void InitDeviceCtx(std::unique_ptr<DeviceCtx>* device_ctx, Stream* stream) const override {}

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

class NopInstructionType final : public InstructionType {
 public:
  NopInstructionType() = default;
  ~NopInstructionType() override = default;

  using stream_type = NopStreamType;

  void Infer(Instruction* instruction) const override { /* do nothing */
  }
  void Compute(Instruction* instruction) const override { UNIMPLEMENTED(); }
};
COMMAND(RegisterInstructionType<NopInstructionType>("Nop"));

void NopStreamType::InitInstructionStatus(const Stream& stream,
                                          InstructionStatusBuffer* status_buffer) const {
  static_assert(sizeof(NaiveInstrStatusQuerier) < kInstructionStatusBufferBytes, "");
  NaiveInstrStatusQuerier::PlacementNew(status_buffer->mut_buffer()->mut_data());
}

void NopStreamType::DeleteInstructionStatus(const Stream& stream,
                                            InstructionStatusBuffer* status_buffer) const {
  // do nothing
}

bool NopStreamType::QueryInstructionStatusDone(const Stream& stream,
                                               const InstructionStatusBuffer& status_buffer) const {
  return NaiveInstrStatusQuerier::Cast(status_buffer.buffer().data())->done();
}

void NopStreamType::Compute(Instruction* instruction) const {
  auto* status_buffer = instruction->mut_status_buffer();
  NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data())->set_done();
}

ObjectMsgPtr<StreamDesc> NopStreamType::MakeStreamDesc(const Resource& resource,
                                                       int64_t this_machine_id) const {
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->mutable_stream_type_id()->__Init__(LookupStreamType4TypeIndex<NopStreamType>());
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(1);
  ret->set_num_streams_per_thread(1);
  ret->set_start_global_device_id(this_machine_id);
  return ret;
}

}  // namespace vm
}  // namespace oneflow
