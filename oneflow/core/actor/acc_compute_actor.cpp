#include "oneflow/core/actor/acc_compute_actor.h"

namespace oneflow {

void AccCompActor::VirtualCompActorInit(const TaskProto& proto) {
  const Shape& one_time_shape = Global<RegstMgr>::Get()
                                    ->RegstDesc4RegstDescId(Name2SoleRegstDescId("one"))
                                    .data_regst_time_shape();
  const Shape& acc_time_shape = Global<RegstMgr>::Get()
                                    ->RegstDesc4RegstDescId(Name2SoleRegstDescId("acc"))
                                    .data_regst_time_shape();
  CHECK_GE(one_time_shape.elem_cnt(), acc_time_shape.elem_cnt());
  AccumulateCompActor::Init(proto, one_time_shape.elem_cnt() / acc_time_shape.elem_cnt(),
                            ColIdOrder::kAscending);
}

REGISTER_ACTOR(TaskType::kAcc, AccCompActor);

}  // namespace oneflow
