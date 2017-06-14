#ifndef ONEFLOW_CORE_ACTOR_BOXING_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_BOXING_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class BoxingActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingActor);
  BoxingActor() = default;
  ~BoxingActor() = default;

  void Init(const TaskProto&) override;
  void ProcessMsg(const ActorMsg&, const ThreadContext&) override;

 private:
  using RDescId2RwMap = HashMap<uint64_t, std::shared_ptr<RegstWarpper>>;
  using RDescId2RwMapPtr = std::unique_ptr<RDescId2RwMap>;

  void WardKernelAndSendMsg(const KernelCtx&);

  // <piece_id, map>
  HashMap<uint64_t, RDescId2RwMapPtr> waiting_in_regst_;
  std::queue<std::pair<uint64_t, RDescId2RwMapPtr>> ready_in_regst_;
  uint64_t in_regst_desc_num_;

};

}  // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_BOXING_ACTOR_H_
