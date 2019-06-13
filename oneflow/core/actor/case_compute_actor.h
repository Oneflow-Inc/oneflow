#ifndef ONEFLOW_CORE_ACTOR_CASE_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_CASE_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class CaseCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CaseCompActor);
  CaseCompActor() = default;
  ~CaseCompActor() override = default;

 protected:
  void VirtualCompActorInit(const TaskProto&) override;
  bool ProducedCtrlRegstValid(int64_t regst_desc_id) const override;
  bool CheckOutputActId(int64_t regst_desc_id) const override;

 private:
  void Act() override;
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;

  int64_t cur_selected_id_;
  HashMap<int64_t, int64_t> regst_desc_id2piece_id_;
  HashMap<int64_t, int64_t> out_bn_id2regst_desc_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_CASE_COMPUTE_ACTOR_H_
