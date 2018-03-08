#include "oneflow/core/actor/decode_compute_actor.h"

namespace oneflow {

void DecodeCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  // data_load_status_.next_col_id = 0;
  // data_load_status_.max_col_id = -1;
  // data_load_status_.next_piece_id = 0;
  // data_load_status_.is_eof = false;
  // OF_SET_MSG_HANDLER(&DecodeCompActor::HandlerWaitToStart);
}

int DecodeCompActor::HandlerWaitToStart(const ActorMsg& msg) {
  // CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
  // ActUntilFail();
  // OF_SET_MSG_HANDLER(&DecodeCompActor::HandlerNormal);
  return 0;
}

int DecodeCompActor::HandlerNormal(const ActorMsg& msg) {
  // CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst()), 0);
  // ActUntilFail();
  // return TrySwitchToZombieOrFinish();
  return 0;
}

void DecodeCompActor::Act() {
  // KernelCtx kernel_ctx = GenDefaultKernelCtx();
  // kernel_ctx.other = &data_load_status_;
  // AsyncLaunchKernel(kernel_ctx, [this](int64_t regst_desc_id) -> Regst* {
  //   return GetCurWriteableRegst(regst_desc_id);
  // });
  // AsyncSendRegstMsgToConsumer([this](Regst* regst) {
  //   regst->set_piece_id(data_load_status_.next_piece_id - 1);
  //   regst->set_col_id(data_load_status_.next_col_id - 1);
  //   regst->set_max_col_id(data_load_status_.max_col_id);
  //   return true;
  // });
}

bool DecodeCompActor::IsReadReady() {
  // bool all_columns_has_read =
  //     data_load_status_.next_col_id > data_load_status_.max_col_id;
  // bool all_piece_has_read =
  //     data_load_status_.is_eof
  //     || data_load_status_.next_piece_id
  //            == RuntimeCtx::Singleton()->total_piece_num();
  // return !all_columns_has_read || !all_piece_has_read;
  return false;
}

REGISTER_ACTOR(kDecode, DecodeCompActor);

}  // namespace oneflow
