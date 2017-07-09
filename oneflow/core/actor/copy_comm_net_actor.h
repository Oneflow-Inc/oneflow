#ifndef ONEFLOW_CORE_ACTOR_COPY_COMM_NET_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_COPY_COMM_NET_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class CopyCommNetActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyCommNetActor);
  CopyCommNetActor() = default;
  ~CopyCommNetActor() = default;

  void Init(const TaskProto&, const ThreadCtx&) override;

 private:
  int HandleNormal(const ActorMsg&) override;
  int HandleWaitUntilNoReadableRegst(const ActorMsg&) override;

  bool IsReadReady() override {
    return piece_id2waiting_in_regst_.find(expected_piece_id())
           != piece_id2waiting_in_regst_.end();
  }
  void Act() override;
  void ProcessEord() override { TODO(); }

  HashMap<int64_t, std::shared_ptr<RegstWrapper>> piece_id2waiting_in_regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_COPY_COMM_NET_ACTOR_H_
