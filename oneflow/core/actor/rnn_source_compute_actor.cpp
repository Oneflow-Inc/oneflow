#include <utility>
#include "oneflow/core/actor/rnn_source_compute_actor.h"

namespace oneflow {

void RnnSourceCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  is_eof_ = false;
  is_final_piece_done_ = false;
  OF_SET_MSG_HANDLER(&RnnSourceCompActor::HandlerWaitToStart);
}

int RnnSourceCompActor::HandlerWaitToStart(const ActorMsg& msg) {
  CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
  ActUntilFail();
  OF_SET_MSG_HANDLER(&RnnSourceCompActor::HandlerNormal);
  return 0;
}

int RnnSourceCompActor::HandlerNormal(const ActorMsg& msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst()), 0);
  ActUntilFail();
  return TrySwitchToZombieOrFinish();
}

void RnnSourceCompActor::Act() {
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  auto ctx = std::make_pair(&data_load_buf_, &is_eof_);
  kernel_ctx.other = static_cast<void*>(&ctx);

  AsyncLaunchKernel(kernel_ctx, [this](int64_t regst_desc_id) -> Regst* {
    return GetCurWriteableRegst(regst_desc_id);
  });
  AsyncSendRegstMsgToConsumer( [this](Regst* regst) { return nullptr; });
  // for CPU actor, there is no async
  if (data_load_buf_.piece_id == RuntimeCtx::Singleton()->total_piece_num() - 1
      && data_load_buf_.col_id == data_load_buf_.max_real_col_num) {
    is_final_piece_done_ = true;
  }
}

bool RnnSourceCompActor::IsReadReady() {
  return !is_eof_ && !is_final_piece_done_;
}

REGISTER_ACTOR(kRnnSource, RnnSourceCompActor);

}  // namespace oneflow
