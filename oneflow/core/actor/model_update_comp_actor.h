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
  int ProcessMsg(const ActorMsg&, const ThreadContext&) override;

 private:
  int HandleBeforeInitKernelCtx(const ActorMsg&, const ThreadContext&);
  int HandleBeforeInitializeModel(const ActorMsg&, const ThreadContext&);
  int HandleBeforeSendInitialModel(const ActorMsg&, const ThreadContext&);
  int HandleUpdateModel(const ActorMsg&, const ThreadContext&);
  int HandleUpdtModelWhenNoReadableRegstMsg(const ActorMsg&, const ThreadContext&);

  void TryWardKernelAndSendMsg();

  CudaStreamHandle cuda_handle_;
  int (MdUpdtCompActor::*cur_msg_handle_)(const ActorMsg&, const ThreadContext&);
  uint64_t model_regst_desc_id_;
  uint64_t model_tmp_regst_desc_id_;
  std::queue<std::shared_ptr<RegstWarpper>> waiting_model_diff_acc_queue_;
  uint64_t next_model_version_id_;

};

}  // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_MODEL_UPDATE_COMP_ACTOR_H_
