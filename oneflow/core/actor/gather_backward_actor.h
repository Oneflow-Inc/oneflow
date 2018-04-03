#ifndef ONEFLOW_CORE_ACTOR_GATHER_BACKWARD_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_GATHER_BACKWARD_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class GatherBackwardActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GatherBackwardActor);
  GatherBackwardActor() = default;
  ~GatherBackwardActor() = default;

  void VirtualActorInit(const TaskProto&) override;

 private:
  int HandlerNormal(const ActorMsg&) override;

  void Act() override;
  bool IsReadReady() override { return !out_diff_regst_.empty(); };
  bool IsReadAlwaysUnReadyFromNow() override;
  void AsyncReturnAllReadableRegst() override;

  void ForEachCurReadableRegst(std::function<void(const Regst*)>) override;

  int64_t in_regst_desc_id_;
  int64_t out_diff_regst_desc_id_;

  bool is_out_diff_eord_;
  int32_t cur_generated_cid_;
  std::queue<Regst*> out_diff_regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_GATHER_BACKWARD_ACTOR_H_
