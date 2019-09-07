#include "oneflow/core/actor/acc_tick_compute_actor.h"

namespace oneflow {

void AccTickCompActor::VirtualCompActorInit(const TaskProto& proto) {
  const Shape& one_time_shape = Global<RegstMgr>::Get()
                                    ->RegstDesc4RegstDescId(Name2SoleRegstDescId("one"))
                                    .data_regst_time_shape();
  const Shape& acc_time_shape = Global<RegstMgr>::Get()
                                    ->RegstDesc4RegstDescId(Name2SoleRegstDescId("acc"))
                                    .data_regst_time_shape();
  CHECK_EQ(one_time_shape.elem_cnt() % acc_time_shape.elem_cnt(), 0);

  acc_cnt_ = 0;
  max_acc_cnt_ = one_time_shape.elem_cnt() / acc_time_shape.elem_cnt();
  OF_SET_MSG_HANDLER(&AccTickCompActor::HandlerNormal);
}

int64_t AccTickCompActor::ActNumForEachOutput(int64_t regst_desc_id) const {
  return regst_desc_id == Name2SoleRegstDescId("acc") ? max_acc_cnt_ : 1;
}

void AccTickCompActor::Act() { acc_cnt_ += 1; }

void AccTickCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  if (acc_cnt_ == max_acc_cnt_) {
    HandleProducedNaiveDataRegstToConsumer();
    acc_cnt_ = 0;
  }
}

REGISTER_ACTOR(TaskType::kAccTick, AccTickCompActor);

}  // namespace oneflow
