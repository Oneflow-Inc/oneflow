#ifndef ONEFLOW_CORE_ACTOR_RNN_BOXING_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_RNN_BOXING_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class RecurrentBoxingActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RecurrentBoxingActor);
  RecurrentBoxingActor() = default;
  ~RecurrentBoxingActor() = default;

  void VirtualActorInit(const TaskProto&) override;

 private:
  int HandlerNormal(const ActorMsg&) override;

  void Act() override;
  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override;
  void AsyncReturnAllReadableRegst() override;

  void TrySetAscendingStatus(const Regst*);

  // <regst_desc_id, regst*>
  HashMap<int64_t, std::queue<Regst*>> readable_regst_;
  HashMap<int64_t, PieceStatus> previous_pst_;
  int64_t readable_regst_cnt_;
  // 0 for unset, 1 for ascending, -1 for descending
  int ascending_status_;
  bool is_eord_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_RNN_BOXING_ACTOR_H_
