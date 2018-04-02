#include "oneflow/core/actor/decode_compute_actor.h"

namespace oneflow {

void DecodeCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  is_in_eord_ = false;
  has_in_regsts_ = true;
  piece_id_ = 0;
  decode_status_.in_regst_ = nullptr;
  decode_status_.cur_col_id_ = 0;
  decode_status_.max_col_id_ = 0;
  OF_SET_MSG_HANDLER(&DecodeCompActor::HandlerWaitToStart);
}

int DecodeCompActor::HandlerWaitToStart(const ActorMsg& msg) {
  OF_SET_MSG_HANDLER(&DecodeCompActor::HandlerNormal);
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
    has_in_regsts_ = false;
    ActUntilFail();
    return 0;
  } else {
    return HandlerNormal(msg);
  }
}

int DecodeCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    is_in_eord_ = true;
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(regst) == -1) {
      if (has_in_regsts_) {
        pending_in_regsts_.push(regst);
      } else {
        UNIMPLEMENTED();
      }
    }
    ActUntilFail();
  } else {
    UNIMPLEMENTED();
  }
  return TrySwitchToZombieOrFinish();
}

void DecodeCompActor::Act() {
  if (decode_status_.in_regst_ == nullptr && has_in_regsts_) {
    decode_status_.in_regst_ = pending_in_regsts_.front();
  }
  CHECK_LE(decode_status_.cur_col_id_, decode_status_.max_col_id_);
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = &decode_status_;
  AsyncLaunchKernel(kernel_ctx, [this](int64_t regst_desc_id) -> Regst* {
    return GetCurWriteableRegst(regst_desc_id);
  });
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    if (has_in_regsts_) {
      regst->set_piece_id(decode_status_.in_regst_->piece_id());
    } else {
      regst->set_piece_id(piece_id_);
    }
    regst->set_col_id(decode_status_.cur_col_id_);
    regst->set_max_col_id(decode_status_.max_col_id_);
    return true;
  });
  if (decode_status_.cur_col_id_ == decode_status_.max_col_id_) {
    if (has_in_regsts_) {
      AsyncSendRegstMsgToProducer(decode_status_.in_regst_);
      pending_in_regsts_.pop();
      decode_status_.in_regst_ = nullptr;
    } else {
      ++piece_id_;
    }
    decode_status_.cur_col_id_ = 0;
    decode_status_.max_col_id_ = 0;
  } else {
    ++decode_status_.cur_col_id_;
  }
}

bool DecodeCompActor::IsReadReady() {
  if (has_in_regsts_) {
    return !pending_in_regsts_.empty();
  } else {
    return piece_id_ < Global<RuntimeCtx>::Get()->total_piece_num();
  }
}

bool DecodeCompActor::IsReadAlwaysUnReadyFromNow() {
  if (has_in_regsts_) {
    return is_in_eord_ && pending_in_regsts_.empty();
  } else {
    return piece_id_ >= Global<RuntimeCtx>::Get()->total_piece_num();
  }
}

void DecodeCompActor::ForEachCurReadableRegst(
    std::function<void(const Regst*)> handler) {
  if (!pending_in_regsts_.empty()) { handler(pending_in_regsts_.front()); }
}

REGISTER_ACTOR(kDecode, DecodeCompActor);

}  // namespace oneflow
