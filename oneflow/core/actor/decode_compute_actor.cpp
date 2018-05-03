#include "oneflow/core/actor/decode_compute_actor.h"

namespace oneflow {

void DecodeCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  piece_id_ = 0;
  decode_status_.in_regst_ = nullptr;
  decode_status_.cur_col_id_ = 0;
  decode_status_.max_col_id_ = 0;
  OF_SET_MSG_HANDLER(&DecodeCompActor::HandlerNormal);
}

void DecodeCompActor::Act(std::function<bool(Regst*)>* IsNaiveAllowedReturnToProducer) {
  if (decode_status_.in_regst_ == nullptr) { decode_status_.in_regst_ = GetNaiveSoleCurReadable(); }
  CHECK_LE(decode_status_.cur_col_id_, decode_status_.max_col_id_);
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = &decode_status_;
  AsyncLaunchKernel(kernel_ctx);
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_piece_id(decode_status_.in_regst_->piece_id());
    regst->set_col_id(decode_status_.cur_col_id_);
    regst->set_max_col_id(decode_status_.max_col_id_);
    return true;
  });
  if (decode_status_.cur_col_id_ == decode_status_.max_col_id_) {
    decode_status_.in_regst_ = nullptr;
    decode_status_.cur_col_id_ = 0;
    decode_status_.max_col_id_ = 0;
  } else {
    ++decode_status_.cur_col_id_;
  }
  *IsNaiveAllowedReturnToProducer = [this](Regst* regst) {
    return decode_status_.in_regst_ == nullptr;
  };
}

REGISTER_ACTOR(kDecode, DecodeCompActor);

}  // namespace oneflow
