#include "oneflow/core/actor/accuracy_accumulate_compute_actor.h"

namespace oneflow {

void AccuracyAccCompActor::VirtualCompActorInit(const TaskProto& proto) {
  const Shape& one_time_shape = Global<RegstMgr>::Get()
                                    ->RegstDesc4RegstDescId(Name2SoleRegstDescId("one"))
                                    .data_regst_time_shape();
  const Shape& acc_time_shape = Global<RegstMgr>::Get()
                                    ->RegstDesc4RegstDescId(Name2SoleRegstDescId("acc"))
                                    .data_regst_time_shape();
  CHECK_GE(one_time_shape.elem_cnt(), acc_time_shape.elem_cnt());
  AccumulateCompActor::Init(
      proto, static_cast<int32_t>(one_time_shape.elem_cnt() / acc_time_shape.elem_cnt()),
      ColIdOrder::kAscending);
}

REGISTER_ACTOR(TaskType::kAccuracyAcc, AccuracyAccCompActor);

}  // namespace oneflow
