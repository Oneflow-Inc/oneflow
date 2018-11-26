#ifndef ONEFLOW_CORE_ACTOR_BOXING_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_BOXING_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class BoxingActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingActor);
  BoxingActor() = default;
  ~BoxingActor() = default;

  void VirtualActorInit(const TaskProto&) override;

 private:
  void NormalProcessNaiveReadableDataRegstMsg(const std::deque<Regst*>&) override;
  void Act() override;
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer();
  void VirtualAsyncSendNaiveConsumedRegstMsgToProducer();
  void TrySetColIdOrder(const Regst*);

  // <regst_desc_id, <pid, cid>>
  std::unique_ptr<HashMap<int64_t, std::pair<int64_t, int32_t>>> previous_pid_cid_;
  ColIdOrder col_id_order_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_BOXING_ACTOR_H_
