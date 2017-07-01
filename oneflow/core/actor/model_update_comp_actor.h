#ifndef ONEFLOW_CORE_ACTOR_MODEL_UPDATE_COMP_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_MODEL_UPDATE_COMP_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class MdUpdtCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MdUpdtCompActor);
  MdUpdtCompActor() = default;
  ~MdUpdtCompActor() = default;

  void Init(const TaskProto&, const ThreadCtx&) override;

 private:
  int HandleBeforeInitDeviceCtx(const ActorMsg&);
  int HandleBeforeInitializeModel(const ActorMsg&);
  int HandleBeforeSendInitialModel(const ActorMsg&);
  int HandleUpdateModel(const ActorMsg&);
  int HandleUpdtModelWhenNoReadableRegstMsg(const ActorMsg&);

  void TryWardKernelAndSendMsg();

  CudaStreamHandle cuda_handle_;
  int64_t model_regst_desc_id_;
  int64_t model_tmp_regst_desc_id_;
  std::queue<std::shared_ptr<RegstWarpper>> waiting_model_diff_acc_queue_;
  int64_t next_model_version_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_MODEL_UPDATE_COMP_ACTOR_H_
