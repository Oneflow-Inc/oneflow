#ifndef ONEFLOW_CORE_ACTOR_ACCUMULATE_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_ACCUMULATE_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class AccumulateCompActor : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AccumulateCompActor);
  AccumulateCompActor() = default;
  virtual ~AccumulateCompActor() = default;

 protected:
  void Init(const TaskProto&, int32_t max_acc_cnt, ColIdOrder order);
  int64_t ActNumForEachOutput(int64_t regst_desc_id) const override;

 private:
  void Act() override;
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;

  ColIdOrder order_;
  std::function<void(DeviceCtx*, void* dst, const void* src, size_t)> cpy_func_;
  int32_t acc_cnt_;
  int32_t max_acc_cnt_;
  int64_t next_piece_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_ACCUMULATE_COMPUTE_ACTOR_H_
