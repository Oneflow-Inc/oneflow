#ifndef ONEFLOW_ACTOR_BOXING_ACTOR_H_
#define ONEFLOW_ACTOR_BOXING_ACTOR_H_

#include "actor/actor.h"

namespace oneflow {

class BoxingActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BoxingActor);
  BoxingActor() = default;
  ~BoxingActor() = default;

  void Init(const TaskProto&) override;
  void ProcessMsg(const ActorMsg&) override;

 private:

  void WardKernelAndSendMsg();

  // <piece_id, <regst_desc_id, subscribed regst>>
  HashMap<uint64_t, std::unique_ptr<HashMap<uint64_t, Regst*>>> waiting_sregst_;
  std::queue<std::pair<uint64_t, std::unique_ptr<HashMap<uint64_t, Regst*>>>> ready_sregst_;
  uint64_t subscribed_regst_desc_num_;
  // <regst_desc_id, produced regst>
  HashMap<uint64_t, std::queue<Regst*>> waiting_pregst_;
  uint64_t waiting_pregst_desc_num_;

};

}  // namespace oneflow

#endif  // ONEFLOW_ACTOR_BOXING_ACTOR_H_
