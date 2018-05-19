#include "oneflow/core/actor/reduce_ag_actor.h"

namespace oneflow {

void ReduceAGActor::VirtualActorInit(const TaskProto& task_proto) {
  pending_in_regsts_.clear();
  for (auto& pair : task_proto.consumed_regst_desc_id()) {
    CHECK_EQ(pair.second.regst_desc_id().size(), 1);
    pending_in_regsts_[pair.second.regst_desc_id().Get(0)] = {};
  }
  processed_regsts_cnt_ = 0;
  in_regsts_eord_cnt_ = 0;
  ready_in_regsts_.clear();
  OF_SET_MSG_HANDLER(&ReduceAGActor::HandlerNormal);
}

bool ReduceAGActor::IsCustomizedReadReady() { return !ready_in_regsts_.empty(); }

bool ReduceAGActor::IsCustomizedReadAlwaysUnReadyFromNow() {
  return in_regsts_eord_cnt_ == pending_in_regsts_.size();
}

void ReduceAGActor::NormalProcessCustomizedEordMsg(const ActorMsg& msg) {
  CHECK(pending_in_regsts_.find(msg.eord_regst_desc_id()) != pending_in_regsts_.end());
  ++in_regsts_eord_cnt_;
}

void ReduceAGActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  Regst* regst = msg.regst();
  if (regst->piece_id() == GetCurPieceId()) {
    CHECK(ready_in_regsts_.emplace(regst->regst_desc_id(), regst).second);
  } else {
    auto pending_in_regsts_it = pending_in_regsts_.find(regst->regst_desc_id());
    CHECK(pending_in_regsts_it != pending_in_regsts_.end());
    pending_in_regsts_it->second.push(regst);
  }
}

void ReduceAGActor::Act() {
  int64_t cur_piece_id = GetCurPieceId();
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = reinterpret_cast<void*>(processed_regsts_cnt_);
  AsyncLaunchKernel(kernel_ctx, [=](int64_t regest_desc_id) -> Regst* {
    auto ready_in_regsts_it = ready_in_regsts_.find(regest_desc_id);
    if (ready_in_regsts_it != ready_in_regsts_.end()) {
      return ready_in_regsts_it->second;
    } else {
      return nullptr;
    }
  });
  processed_regsts_cnt_ += ready_in_regsts_.size();
  for (auto pair : ready_in_regsts_) { AsyncSendRegstMsgToProducer(pair.second); }
  ready_in_regsts_.clear();
  if (GetCurPieceId() == cur_piece_id + 1) {
    AsyncSendRegstMsgToConsumer([&](Regst* regst) {
      regst->set_piece_id(cur_piece_id);
      return true;
    });
    for (auto& pair : pending_in_regsts_) {
      if (pair.second.empty()) { continue; }
      Regst* regst = pair.second.front();
      CHECK_EQ(regst->piece_id(), cur_piece_id + 1);
      CHECK(ready_in_regsts_.emplace(regst->regst_desc_id(), regst).second);
      pair.second.pop();
    }
  }
}

REGISTER_ACTOR(TaskType::kReduceGather, ReduceAGActor);
REGISTER_ACTOR(TaskType::kReduceAdd, ReduceAGActor);

}  // namespace oneflow
