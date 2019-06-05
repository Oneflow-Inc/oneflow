#ifndef ONEFLOW_CORE_ACTOR_NORMAL_BACKWARD_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_NORMAL_BACKWARD_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class NormalBackwardCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(NormalBackwardCompActor);
  NormalBackwardCompActor() = default;
  ~NormalBackwardCompActor() = default;

  void VirtualCompActorInit(const TaskProto&) override;

 private:
  void ForEachCurCustomizedReadableRegst(std::function<void(const Regst*)>) const override;
  void NormalProcessNaiveReadableDataRegstMsg(const std::deque<Regst*>&) override;
  void NormalProcessCustomizedReadableRegstMsg(const ActorMsg&) override;
  void Act() override;
  bool IsCustomizedReadReady() const override;
  void AsyncReturnAllCustomizedReadableRegst() override;
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedConsumedRegstDescName()
      override {
    return std::make_pair(RegstNameType::kNaive,
                          HashSet<std::string>{"activation", "data_tmp", "out", "out_diff", "in"});
  }
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;
  void AsyncSendCustomizedConsumedRegstMsgToProducer() override;

  void AsyncReturnModelRegstUntilModelVersionIdEqual(int64_t model_version_id);
  void AsyncReturnModelRegstUntilLastPieceIdGreaterThan(int64_t piece_id);

  int64_t actual_num_of_piece_in_batch_;
  int64_t any_out_diff_regst_desc_id_;
  int64_t model_regst_desc_id_;
  int64_t const_buf_regst_desc_id_;
  int64_t const_model_regst_desc_id_;
  std::queue<Regst*> model_regst_queue_;
  Regst* const_model_regst_;
  Regst* const_buf_regst_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_NORMAL_BACKWARD_COMPUTE_ACTOR_H_
