#include "oneflow/core/actor/accuracy_accumulate_compute_actor.h"

namespace oneflow {

void AccuracyAccCompActor::VirtualCompActorInit(const TaskProto& proto) {
  const Shape& one_time_shape = Global<RegstMgr>::Get()
                                    ->RegstDesc4RegstDescId(Name2SoleRegstDescId("one"))
                                    .data_regst_time_shape();
  CHECK_GE(one_time_shape.NumAxes(), 2);
  CHECK_EQ(one_time_shape.At(0), Global<JobDesc>::Get()->TotalBatchNum());
  CHECK_EQ(one_time_shape.At(1), Global<JobDesc>::Get()->NumOfPiecesInBatch());
  AccumulateCompActor::Init(
      proto,
      static_cast<int32_t>(one_time_shape.Count(2)
                           * Global<JobDesc>::Get()->PieceNumOfPrintAccuracy()),
      ColIdOrder::kAscending);
}

REGISTER_ACTOR(TaskType::kAccuracyAcc, AccuracyAccCompActor);

}  // namespace oneflow
