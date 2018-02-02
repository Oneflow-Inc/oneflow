#ifndef ONEFLOW_CORE_ACTOR_BOXING_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_BOXING_ACTOR_H_

#include "oneflow/core/actor/actor.h"
#include "oneflow/core/actor/naive_readable_register_manager.h"

namespace oneflow {

class BoxingActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingActor);
  BoxingActor() = default;
  ~BoxingActor() = default;

  void VirtualActorInit(const TaskProto&) override;

 private:
  int HandlerNormal(const ActorMsg&) override;

  void Act() override;
  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override;
  void AsyncReturnAllReadableRegst() override;

  void ForEachCurReadableRegst(std::function<void(const Regst*)>) override;

  void TrySetColIdOrder(const Regst*);

  NaiveReadableRegstMgr readable_regst_mgr_;
  // <regst_desc_id, <pid, cid>>
  HashMap<int64_t, std::pair<int64_t, int32_t>>* previous_pid_cid_;
  ColIdOrder col_id_order_;
  bool is_eord_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_BOXING_ACTOR_H_
