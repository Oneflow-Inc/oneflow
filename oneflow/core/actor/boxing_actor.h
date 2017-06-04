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
  void ProcessMsg(const ActorMsg&) override;

 private:

  void WardKernelAndSendMsg();

  // <piece_id, <regst_desc_id, regst>>
  HashMap<uint64_t, std::unique_ptr<HashMap<uint64_t, Regst*>>> waiting_in_regst_;
  std::queue<std::pair<uint64_t, std::unique_ptr<HashMap<uint64_t, Regst*>>>> ready_in_regst_;
  uint64_t in_regst_desc_num_;
  // <regst_desc_id, regst>
  HashMap<uint64_t, std::queue<Regst*>> writeable_out_regst_;
  uint64_t writeable_out_regst_desc_num_;
  HashMap<Regst*, int64_t> out_regst2reading_cnt_;
  // 
  Regst* middle_regst_;

};

}  // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_BOXING_ACTOR_H_
