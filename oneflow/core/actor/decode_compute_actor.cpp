#include "oneflow/core/actor/decode_compute_actor.h"

namespace oneflow {

void DecodeCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  is_in_eord_ = false;
  decode_status_.in_regst_ = nullptr;
  decode_status_.cur_col_id_ = 0;
  decode_status_.max_col_id_ = 0;
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
    ActUntilFail();
  } else {
    UNIMPLEMENTED();
  }
  return TrySwitchToZombieOrFinish();
}

void DecodeCompActor::Act() {
  if (decode_status_.in_regst_ == nullptr) {
    decode_status_.in_regst_ = pending_in_regsts_.front();
  }
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = &decode_status_;
  AsyncLaunchKernel(kernel_ctx, [this](int64_t regst_desc_id) -> Regst* {
    return GetCurWriteableRegst(regst_desc_id);
  });
  CHECK_GT(decode_status_.max_col_id_, 0);
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_piece_id(decode_status_.in_regst_->piece_id());
    regst->set_col_id(decode_status_.cur_col_id_);
    regst->set_max_col_id(decode_status_.max_col_id_);
    return true;
  });
  if (decode_status_.cur_col_id_ == decode_status_.max_col_id_) {
    AsyncSendRegstMsgToProducer(decode_status_.in_regst_);
    pending_in_regsts_.pop();
    decode_status_.in_regst_ = nullptr;
    decode_status_.cur_col_id_ = 0;
    decode_status_.max_col_id_ = 0;
  }
  ++decode_status_.cur_col_id_;
}

bool DecodeCompActor::IsReadReady() { return !pending_in_regsts_.empty(); }

bool DecodeCompActor::IsReadAlwaysUnReadyFromNow() {
  return is_in_eord_ && pending_in_regsts_.empty();
}

void DecodeCompActor::ForEachCurReadableRegst(
    std::function<void(const Regst*)> handler) {
  handler(pending_in_regsts_.front());
}

REGISTER_ACTOR(kDecode, DecodeCompActor);

}  // namespace oneflow
