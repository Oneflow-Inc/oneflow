#ifndef ONEFLOW_CORE_ACTOR_MODEL_UPDATE_COMP_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_MODEL_UPDATE_COMP_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class MdUpdtCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdUpdtCompActor);
  MdUpdtCompActor() = default;
  ~MdUpdtCompActor() = default;

  void Init(const TaskProto&) override;
  void ProcessMsg(const ActorMsg&, const ThreadContext&) override;

 private:
  void HandleBeforeInitKernelCtx(const ActorMsg&, const ThreadContext&);
  void HandleBeforeInitializeModel(const ActorMsg&, const ThreadContext&);
  void HandleBeforeSendInitialModel(const ActorMsg&, const ThreadContext&);
  void HandleForUpdateModel(const ActorMsg&, const ThreadContext&);

  void ProcessRegstFromMsg(std::shared_ptr<RegstWarpper>, const ThreadContext&);

  void (MdUpdtCompActor::*cur_handle_)(const ActorMsg&, const ThreadContext&);
  uint64_t model_regst_desc_id_;
  uint64_t model_tmp_regst_desc_id_;
  std::queue<std::shared_ptr<RegstWarpper>> waiting_model_diff_acc_queue_;

};

}  // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_MODEL_UPDATE_COMP_ACTOR_H_
