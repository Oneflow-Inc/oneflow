#include "oneflow/core/actor/pack_compute_actor.h"

namespace oneflow {

void PackCompActor::VirtualCompActorInit(const TaskProto& proto) {
  const Shape& in_time_shape = Global<RegstMgr>::Get()
                                   ->RegstDesc4RegstDescId(Name2SoleRegstDescId("in"))
                                   .data_regst_time_shape();
  total_pack_num_ = in_time_shape.At(in_time_shape.NumAxes() - 1);
  act_num_cnt_ = 0;
  cur_piece_id_ = 0;
  OF_SET_MSG_HANDLER(&PackCompActor::HandlerNormal);
}

void PackCompActor::Act() {
  KernelCtx ctx = GenDefaultKernelCtx();
  std::pair<size_t, size_t> other_val = std::make_pair(act_num_cnt_, total_pack_num_);
  ctx.other = static_cast<void*>(&other_val);
  AsyncLaunchKernel(ctx);
  act_num_cnt_ += 1;
}

void PackCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  if (act_num_cnt_ == total_pack_num_) {
    HandleProducedNaiveDataRegstToConsumer([this](Regst* regst) {
      regst->set_piece_id(cur_piece_id_);
      return true;
    });
    act_num_cnt_ = 0;
    cur_piece_id_ += 1;
  }
}

}  // namespace oneflow
