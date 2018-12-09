#include "oneflow/core/actor/decode_in_stream_actor.h"

namespace oneflow {

void DecodeInStreamActor::VirtualCompActorInit(const TaskProto& task_proto) {
  OF_SET_MSG_HANDLER(&DecodeInStreamActor::HandlerWaitToStart);
}

void DecodeInStreamActor::Act() {
  Regst* regst = GetNaiveCurWriteable("out");
  regst->set_piece_id(piece_id_++);

  PredictParams records;
  Global<OFServing>::Get()->InChannel().Receive(&records);

  KernelCtx ctx = GenDefaultKernelCtx();
  ctx.other = &records;

  AsyncLaunchKernel(ctx);
}

bool DecodeInStreamActor::IsCustomizedReadReady() {
  bool r = Global<OFServing>::Get()->IsEof();
  return r;
}

int DecodeInStreamActor::HandlerWaitToStart(const ActorMsg& msg) {
  CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
  OF_SET_MSG_HANDLER(&DecodeInStreamActor::HandlerNormal);
  return ProcessMsg(msg);
}

REGISTER_ACTOR(kDecodeInStream, DecodeInStreamActor);
}  // namespace oneflow
