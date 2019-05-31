#include "oneflow/core/actor/loss_accumulate_compute_actor.h"

namespace oneflow {

void LossAccCompActor::VirtualCompActorInit(const TaskProto& proto) {
  const Shape& one_time_shape = Global<RegstMgr>::Get()
                                    ->RegstDesc4RegstDescId(Name2SoleRegstDescId("one"))
                                    .data_regst_time_shape();
  CHECK_GE(one_time_shape.NumAxes(), 2);
  CHECK_EQ(one_time_shape.At(0), job_desc().TotalBatchNum());
  CHECK_EQ(one_time_shape.At(1), job_desc().NumOfPiecesInBatch());
  AccumulateCompActor::Init(
      proto, static_cast<int32_t>(one_time_shape.Count(2) * job_desc().PieceNumOfPrintLoss()),
      ColIdOrder::kAscending);
}

REGISTER_ACTOR(TaskType::kLossAcc, LossAccCompActor);

}  // namespace oneflow
