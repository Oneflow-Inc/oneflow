#ifndef ONEFLOW_CORE_ACTOR_UNPACK_COMPUTE_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_UNPACK_COMPUTE_ACTOR_H_

#include "oneflow/core/actor/compute_actor.h"

namespace oneflow {

class UnpackCompActor final : public CompActor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UnpackCompActor);
  UnpackCompActor() = default;
  ~UnpackCompActor() = default;

 private:
  void VirtualCompActorInit(const TaskProto& proto) override;
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedConsumedRegstDescName()
      override {
    return {RegstNameType::kCustomized, {"in"}};
  }
  void NormalProcessCustomizedEordMsg(const ActorMsg&) override;
  void NormalProcessCustomizedReadableRegstMsg(const ActorMsg&) override;
  bool IsCustomizedReadReady() override;
  bool IsCustomizedReadAlwaysUnReadyFromNow() override;
  void AsyncReturnAllCustomizedReadableRegst() override;

  void Act() override;
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;
  void AsyncSendCustomizedConsumedRegstMsgToProducer() override;
  void VirtualAsyncSendNaiveConsumedRegstMsgToProducer() override;

  size_t total_unpack_num_;
  size_t act_num_cnt_;
  size_t cur_piece_id_;
  bool handle_pack_bw_;

  int64_t in_regst_desc_id_;
  std::queue<Regst*> in_regsts_;
  bool is_in_eord_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_UNPACK_COMPUTE_ACTOR_H_
