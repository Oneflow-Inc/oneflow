#include "oneflow/core/vm/stream.msg.h"
#include "oneflow/core/vm/thread_ctx.msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace vm {

void Stream::__Init__(ThreadCtx* thread_ctx, const StreamId& stream_id) {
  set_thread_ctx(thread_ctx);
  mut_stream_id()->CopyFrom(stream_id);
  const auto& stream_type = thread_ctx->stream_rt_desc().stream_type();
  stream_type.InitDeviceCtx(mut_device_ctx(), this);
}

int64_t Stream::machine_id() const {
  return parallel_id() / thread_ctx().stream_rt_desc().stream_desc().num_streams_per_machine();
}

ObjectMsgPtr<InstrChain> Stream::NewInstrChain(InstructionMsg* instr_msg) {
  if (free_chain_list().empty()) {
    return ObjectMsgPtr<InstrChain>::NewFrom(mut_allocator(), instr_msg, this);
  }
  ObjectMsgPtr<InstrChain> instr_chain = mut_free_chain_list()->PopFront();
  instr_chain->__Init__(instr_msg, this);
  return instr_chain;
}

void Stream::DeleteInstrChain(ObjectMsgPtr<InstrChain>&& instr_chain) {
  CHECK(instr_chain->is_pending_chain_link_empty());
  CHECK(instr_chain->is_instr_chain_link_empty());
  CHECK_EQ(instr_chain->ref_cnt(), 1);
  auto* instr_chain_ptr = instr_chain.Mutable();
  mut_free_chain_list()->EmplaceBack(std::move(instr_chain));
  instr_chain_ptr->__Delete__();
}

}  // namespace vm
}  // namespace oneflow
