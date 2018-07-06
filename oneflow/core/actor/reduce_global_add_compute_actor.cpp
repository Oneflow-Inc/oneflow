#include "oneflow/core/actor/reduce_global_add_compute_actor.h"

namespace oneflow {

void ReduceGlobalAddCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    CHECK_EQ(1, pair.second.regst_desc_id_size());
    int64_t regst_desc_id = pair.second.regst_desc_id().Get(0);
    CHECK(readable_regsts_.emplace(regst_desc_id, std::queue<Regst*>()).second);
    CHECK(unprocessed_regst_desc_id_.emplace(regst_desc_id).second);
  }
  cur_processed_regst_desc_id_ = -1;
  OF_SET_MSG_HANDLER(&ReduceGlobalAddCompActor::HandlerNormal);
}

void ReduceGlobalAddCompActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  Regst* regst = msg.regst();
  int regst_desc_id = regst->regst_desc_id();
  CHECK(readable_regsts_.find(regst_desc_id) != readable_regsts_.end());
  readable_regsts_.at(regst_desc_id).push(regst);
}

bool ReduceGlobalAddCompActor::IsCustomizedReadReady() {
  CHECK_EQ(-1, cur_processed_regst_desc_id_);
  for (const auto& pair : readable_regsts_) {
    if (pair.second.empty()) { continue; }
    if (unprocessed_regst_desc_id_.find(pair.first) != unprocessed_regst_desc_id_.end()) {
      cur_processed_regst_desc_id_ = pair.first;
      return true;
    }
  }
  return false;
}

void ReduceGlobalAddCompActor::ForEachCurCustomizedReadableRegst(
    std::function<void(const Regst*)> handler) const {
  handler(readable_regsts_.at(cur_processed_regst_desc_id_).front());
}

void ReduceGlobalAddCompActor::Act() {
  Regst* cur_regst = readable_regsts_.at(cur_processed_regst_desc_id_).front();
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = &cur_processed_regst_desc_id_;
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* { return cur_regst; });

  readable_regsts_.at(cur_processed_regst_desc_id_).pop();
  cur_processed_regst_desc_id_ = -1;
  unprocessed_regst_desc_id_.erase(cur_processed_regst_desc_id_);
  if (unprocessed_regst_desc_id_.empty()) {
    AsyncSendRegstMsgToConsumer([&](Regst* regst) {
      regst->set_piece_id(cur_regst->piece_id());
      return true;
    });
    for (const auto& pair : readable_regsts_) { unprocessed_regst_desc_id_.insert(pair.first); }
  }
  AsyncSendRegstMsgToProducer(cur_regst);
}

void ReduceGlobalAddCompActor::AsyncReturnAllCustomizedReadableRegst() {
  CHECK(readable_regsts_.empty());
  CHECK_EQ(-1, cur_processed_regst_desc_id_);
}

REGISTER_ACTOR(TaskType::kReduceGlobalAdd, ReduceGlobalAddCompActor);

}  // namespace oneflow
