#ifndef ONEFLOW_CORE_ACTOR_MULTI_RING_ALL_REDUCE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_MULTI_RING_ALL_REDUCE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class MultiRingAllReduceActor : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(MultiRingAllReduceActor);
  MultiRingAllReduceActor() = default;
  ~MultiRingAllReduceActor() override = default;

 protected:
  void VirtualActorInit(const TaskProto&) override;
  int64_t ActNumForEachOutput(int64_t regst_desc_id) const override;
  bool ProducedCtrlRegstValid(int64_t regst_desc_id) const override;

 private:
  void Act() override;
  void NormalProcessCustomizedReadableRegstMsg(const ActorMsg&) override;
  void UpdtStateAsCustomizedProducedRegst(Regst* regst) override;
  void ForEachCurCustomizedReadableRegst(std::function<void(const Regst*)>) const override;
  bool IsCustomizedReadReady() const override;
  bool IsCustomizedWriteReady() const override;
  void NormalProcessCustomizedEordMsg(const ActorMsg&) override;
  bool IsCustomizedReadAlwaysUnReadyFromNow() const override;
  void AsyncReturnAllCustomizedReadableRegst() override;
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedConsumedRegstDescName()
      override {
    return std::make_pair(RegstNameType::kNaive, HashSet<std::string>{});
  }
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedProducedRegstDescName()
      override {
    return std::make_pair(RegstNameType::kNaive, HashSet<std::string>{});
  }
  void AsyncSendCustomizedProducedRegstMsgToConsumer() override;
  void AsyncSendCustomizedConsumedRegstMsgToProducer() override;
  bool CheckOutputActId(int64_t regst_desc_id) const override;
  void SetKernelCtxOther(void** other);

  RegstSlot consumed_rs_;
  RegstSlot produced_rs_;
  int64_t num_steps_ = -1;
  int64_t current_step_ = -1;
  int64_t in_regst_desc_id_ = -1;
  int64_t out_regst_desc_id_ = -1;
  std::vector<int64_t> recv_regst_desc_id_;
  std::vector<int64_t> send_regst_desc_id_;
  std::vector<int64_t> send_regst_piece_id_;
  bool in_regst_eord_ = false;
  int64_t current_ring_id_ = -1;
  int64_t num_rings_ = -1;
  std::pair<int64_t, int64_t> other_ctx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_MULTI_RING_ALL_REDUCE_ACTOR_H_
