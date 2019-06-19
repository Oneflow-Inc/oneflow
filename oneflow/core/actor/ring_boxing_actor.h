#ifndef ONEFLOW_CORE_ACTOR_RING_BOXING_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_RING_BOXING_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class RingBoxingActor : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(RingBoxingActor);
  RingBoxingActor() = default;
  ~RingBoxingActor() override = default;

 protected:
  void VirtualActorInit(const TaskProto&) override;
  int64_t ActNumForEachOutput(int64_t regst_desc_id) const override;
  bool ProducedCtrlRegstValid(int64_t regst_desc_id) const override;

 private:
  void Act() override;
  void NormalProcessCustomizedReadableRegstMsg(const ActorMsg&) override;
  void ForEachCurCustomizedReadableRegst(std::function<void(const Regst*)>) const override;
  bool IsCustomizedReadReady() const override;
  void NormalProcessCustomizedEordMsg(const ActorMsg&) override;
  bool IsCustomizedReadAlwaysUnReadyFromNow() const override;
  void AsyncReturnAllCustomizedReadableRegst() override;
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedConsumedRegstDescName()
      override {
    return std::make_pair(RegstNameType::kNaive, HashSet<std::string>{});
  }
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;
  void AsyncSendCustomizedConsumedRegstMsgToProducer() override;
  bool CheckOutputActId(int64_t regst_desc_id) const override;
  void SetKernelCtxOther(void** other);

  RegstSlot consumed_rs_;
  int64_t total_num_steps_ = -1;
  int64_t current_step_ = -1;
  int64_t in_regst_desc_id_ = -1;
  int64_t out_regst_desc_id_ = -1;
  int64_t recv_regst_desc_id_ = -1;
  int64_t send_regst_desc_id_ = -1;
  int64_t send_regst_piece_id_ = -1;
  bool in_regst_eord_ = false;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_RING_BOXING_ACTOR_H_
