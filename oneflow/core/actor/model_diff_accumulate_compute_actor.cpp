#include "oneflow/core/actor/model_diff_accumulate_compute_actor.h"

namespace oneflow {

void MdDiffAccCompActor::VirtualCompActorInit(const TaskProto& proto) {
  const Shape& one_time_shape = Global<RegstMgr>::Get()
                                    ->RegstDesc4RegstDescId(Name2SoleRegstDescId("one"))
                                    .data_regst_time_shape();
  const Shape& acc_time_shape = Global<RegstMgr>::Get()
                                    ->RegstDesc4RegstDescId(Name2SoleRegstDescId("acc"))
                                    .data_regst_time_shape();
  CHECK_EQ(one_time_shape.At(0), job_desc().TotalBatchNum());
  CHECK_EQ(one_time_shape.At(1), job_desc().NumOfPiecesInBatch());
  CHECK_EQ(acc_time_shape.At(0), job_desc().TotalBatchNum());
  CHECK_EQ(acc_time_shape.NumAxes(), 1);
  AccumulateCompActor::Init(proto, static_cast<int32_t>(one_time_shape.Count(1)),
                            ColIdOrder::kDescending);
}

REGISTER_ACTOR(TaskType::kMdDiffAcc, MdDiffAccCompActor);

}  // namespace oneflow
