#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/vm/nop_stream_type.h"
#include "oneflow/core/vm/instruction.msg.h"
#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/vm/naive_instruction_status_querier.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/resource.pb.h"

namespace oneflow {
namespace vm {

namespace {}  // namespace

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

const StreamTypeId NopStreamType::kStreamTypeId;

ObjectMsgPtr<InstructionMsg> NopStreamType::Nop() const {
  auto instr_msg = ObjectMsgPtr<InstructionMsg>::New();
  auto* instr_type_id = instr_msg->mutable_instr_type_id();
  instr_type_id->set_stream_type_id(kStreamTypeId);
  instr_type_id->set_opcode(0);
  return instr_msg;
}

void NopStreamType::Run(InstrChain* instr_chain) const {
  auto* status_buffer = instr_chain->mut_status_buffer();
  NaiveInstrStatusQuerier::MutCast(status_buffer->mut_buffer()->mut_data())->set_done();
}

ObjectMsgPtr<StreamDesc> NopStreamType::MakeRemoteStreamDesc(const Resource& resource,
                                                             int64_t this_machine_id) const {
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->set_stream_type_id(kStreamTypeId);
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(1);
  ret->set_num_streams_per_thread(1);
  ret->set_start_parallel_id(this_machine_id);
  return ret;
}

ObjectMsgPtr<StreamDesc> NopStreamType::MakeLocalStreamDesc(const Resource& resource) const {
  auto ret = ObjectMsgPtr<StreamDesc>::New();
  ret->set_stream_type_id(kStreamTypeId);
  ret->set_num_machines(1);
  ret->set_num_streams_per_machine(1);
  ret->set_num_streams_per_thread(1);
  ret->set_start_parallel_id(0);
  return ret;
}

COMMAND(RegisterStreamType<NopStreamType>());
COMMAND(RegisterInstrTypeId<NopStreamType>("Nop", 0, kRemote));
COMMAND(RegisterInstrTypeId<NopStreamType>("LocalNop", 0, kLocal));

}  // namespace vm
}  // namespace oneflow
