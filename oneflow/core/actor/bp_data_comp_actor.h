#ifndef ONEFLOW_CORE_ACTOR_BP_DATA_COMP_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_BP_DATA_COMP_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class BpDataCompActor final : public Actor {
public:
  OF_DISALLOW_COPY_AND_MOVE(BpDataCompActor);
  BpDataCompActor() = default;
  ~BpDataCompActor() = default;

  void Init(const TaskProto&, const ThreadCtx&) override;
  int ProcessMsg(const ActorMsg&) override;

private:
  int HandleInitDeviceCtx(const ActorMsg&);
  int HandleBpComp(const ActorMsg&);
  int HandleBpCompWhenNoReadableRegstMsg(const ActorMsg&);
  int HandleWaitUntilReadingCntEqualZero(const ActorMsg&);

  bool IsReadReady();
  void TryWardKernelAndSendMsg();

  CudaStreamHandle cuda_handle_;
  int (BpDataCompActor::*cur_msg_handle_)(const ActorMsg&);
  int num_of_read_empty_;
  int num_of_eord_;
  uint64_t expected_model_version_id_;
  uint64_t model_regst_desc_id_;
  uint64_t model_tmp_regst_desc_id_;
  uint64_t activation_regst_desc_id_;
  uint64_t data_tmp_regst_desc_id_;
  // <regst_desc_id, queue<regst_wp>>
  HashMap<uint64_t, std::queue<std::shared_ptr<RegstWarpper>>> read_regst_;
};

}  // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_BP_DATA_COMP_ACTOR_H_
