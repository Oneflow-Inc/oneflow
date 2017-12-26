#ifndef ONEFLOW_CORE_ACTOR_RNN_BOXING_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_RNN_BOXING_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class RnnBoxingActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RnnBoxingActor);
  RnnBoxingActor() = default;
  ~RnnBoxingActor() = default;

  void VirtualActorInit(const TaskProto&) override;

 private:
  int HandlerNormal(const ActorMsg&) override;

  void Act() override;
  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override;
  void AsyncReturnAllReadableRegst() override;

  // <pid, <regst_desc_id, regst*>>
  std::map<int64_t, HashMap<int64_t, std::queue<Regst*>>> readable_regst_;
  std::map<int64_t, int64_t> readable_regst_cnt_;
  int64_t num_of_consumed_;
  int64_t cur_processing_pid_;
  bool is_ascending_;
  bool is_eord_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_RNN_BOXING_ACTOR_H_
