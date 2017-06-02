#ifndef ONEFLOW_ACTOR_BOXING_ACTOR_H_
#define ONEFLOW_ACTOR_BOXING_ACTOR_H_

#include "oneflow/core/actor/actor.h"

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

  // <piece_id, <regst_desc_id, regst>>
  HashMap<uint64_t, std::unique_ptr<HashMap<uint64_t, Regst*>>> waiting_in_regst_;
  std::queue<std::pair<uint64_t, std::unique_ptr<HashMap<uint64_t, Regst*>>>> ready_in_regst_;
  uint64_t in_regst_desc_num_;
  // <regst_desc_id, regst>
  HashMap<uint64_t, std::queue<Regst*>> waiting_out_regst_;
  uint64_t waiting_out_regst_desc_num_;
  // 
  Regst* middle_regst_;

};

}  // namespace oneflow

#endif  // ONEFLOW_ACTOR_BOXING_ACTOR_H_
