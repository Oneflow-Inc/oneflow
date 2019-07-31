#ifndef ONEFLOW_CORE_ACTOR_CUDA_RING_ALL_REDUCE_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_CUDA_RING_ALL_REDUCE_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class CudaRingAllReduceCompActor : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaRingAllReduceCompActor);
  CudaRingAllReduceCompActor() = default;
  ~CudaRingAllReduceCompActor() override = default;

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

  int64_t num_link_ = -1;
  int64_t slice_factor_ = -1;
  int64_t num_step_ = -1;
  int64_t current_step_id_ = -1;
  int64_t current_slice_id_ = -1;
  int64_t in_regst_desc_id_ = -1;
  int64_t out_regst_desc_id_ = -1;
  HashSet<int64_t> send_regst_desc_ids_;
  HashSet<int64_t> recv_regst_desc_ids_;
  RegstSlot consumed_rs_;
  int64_t send_regst_piece_id_ = -1;
  bool in_regst_eord_ = false;
  std::pair<int64_t, int64_t> other_ctx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_CUDA_RING_ALL_REDUCE_COMPUTE_ACTOR_H_
