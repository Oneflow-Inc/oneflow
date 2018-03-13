#include "oneflow/core/actor/decode_compute_actor.h"

namespace oneflow {

void DecodeCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  is_in_eord_ = false;
  OF_SET_MSG_HANDLER(&DecodeCompActor::HandlerNormal);
}

int DecodeCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    is_in_eord_ = true;
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(regst) == -1) {
      pending_in_regsts_.push(regst);
    }
  } else {
    UNIMPLEMENTED();
  }
  return TrySwitchToZombieOrFinish();
}

void DecodeCompActor::Act() {
  Regst* in_regst = pending_in_regsts_.front();
  pending_in_regsts_.pop();
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = in_regst;
  AsyncLaunchKernel(kernel_ctx, [this](int64_t regst_desc_id) -> Regst* {
    return GetCurWriteableRegst(regst_desc_id);
  });
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_piece_id(in_regst->piece_id());
    return true;
  });
  AsyncSendRegstMsgToProducer(in_regst);
}

bool DecodeCompActor::IsReadReady() { return !pending_in_regsts_.empty(); }

bool DecodeCompActor::IsReadAlwaysUnReadyFromNow() {
  return is_in_eord_ && pending_in_regsts_.empty();
}

REGISTER_ACTOR(kDecode, DecodeCompActor);

}  // namespace oneflow
