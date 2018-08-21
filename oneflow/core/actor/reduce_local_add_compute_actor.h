#ifndef ONEFLOW_CORE_ACTOR_REDUCE_LOCAL_ADD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_REDUCE_LOCAL_ADD_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/input_wise_compute_actor.h"

namespace oneflow {

class ReduceLocalAddCompActor final : public InputWiseCompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReduceLocalAddCompActor);
  ReduceLocalAddCompActor() = default;
  ~ReduceLocalAddCompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto& proto) override;
  void SetKernelCtxOther(void** other) override;
  void VirtualUpdateMemberStatusAfterAct() override;
  void VirtualUpdateMemberStatusAfterSendRegstMsgToConsumer() override;

  std::vector<bool> out_blob_init_status_;
  int64_t cur_out_blob_id_;
  HashSet<int64_t> inplace_blob_ids_;
  std::tuple<int64_t, int64_t, bool, bool> other_val_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_REDUCE_LOCAL_ADD_COMPUTE_ACTOR_H_
