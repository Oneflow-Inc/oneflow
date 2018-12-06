#include "oneflow/core/actor/out_stream_actor.h"

namespace oneflow {

void OutStreamCompActor::VirtualSinkCompActorInit(const TaskProto&) {}

void OutStreamCompActor::Act() {
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  std::pair<int64_t, std::string> data = {-1, ""};
  kernel_ctx.other = &data;
  AsyncLaunchKernel(kernel_ctx);

  Global<OFServing>::Get()->SendMsg(data);
}

void OutStreamCompActor::AsyncSendCustomizedProducedRegstMsgToConsumer() {}

REGISTER_ACTOR(TaskType::kOutStream, OutStreamCompActor);
}  // namespace oneflow