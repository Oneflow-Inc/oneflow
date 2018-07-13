#include "oneflow/core/actor/input_wise_compute_actor.h"

namespace oneflow {

void InputWiseCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  for (const auto& pair : task_proto.exec_sequence().exec_node().Get(0).bn_in_op2regst_desc_id()) {
    CHECK(regst_desc_id2bn_in_op_.emplace(pair.second, pair.first).second);
  }
  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    CHECK_EQ(1, pair.second.regst_desc_id_size());
    int64_t regst_desc_id = pair.second.regst_desc_id().Get(0);
    CHECK(readable_regsts_.emplace(regst_desc_id, std::queue<Regst*>()).second);
    CHECK(regst_desc_id2is_processed_.emplace(regst_desc_id, false).second);
  }
  cur_processed_regst_desc_id_ = -1;
  readable_regst_desc_cnt_ = 0;
  processed_regst_desc_id_cnt_ = 0;
  OF_SET_MSG_HANDLER(&InputWiseCompActor::HandlerNormal);
}

void InputWiseCompActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  Regst* regst = msg.regst();
  int regst_desc_id = regst->regst_desc_id();
  CHECK(readable_regsts_.find(regst_desc_id) != readable_regsts_.end());
  std::queue<Regst*>& regst_q = readable_regsts_.at(regst_desc_id);
  if (regst_q.empty()) { readable_regst_desc_cnt_ += 1; }
  regst_q.push(regst);
}

bool InputWiseCompActor::IsCustomizedReadReady() {
  CHECK_EQ(-1, cur_processed_regst_desc_id_);
  for (const auto& pair : readable_regsts_) {
    if (pair.second.empty()) { continue; }
    if (regst_desc_id2is_processed_.at(pair.first) == false) {
      cur_processed_regst_desc_id_ = pair.first;
      return true;
    }
  }
  return false;
}

void InputWiseCompActor::ForEachCurCustomizedReadableRegst(
    std::function<void(const Regst*)> handler) const {
  handler(readable_regsts_.at(cur_processed_regst_desc_id_).front());
}

void InputWiseCompActor::Act() {
  std::queue<Regst*>& regst_q = readable_regsts_.at(cur_processed_regst_desc_id_);
  Regst* cur_regst = regst_q.front();
  KernelCtx kernel_ctx = GenDefaultKernelCtx();

  std::pair<std::string, bool> other_val(regst_desc_id2bn_in_op_.at(cur_processed_regst_desc_id_),
                                         processed_regst_desc_id_cnt_ == 0);
  kernel_ctx.other = &other_val;
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    CHECK_EQ(cur_processed_regst_desc_id_, regst_desc_id);
    return cur_regst;
  });

  regst_q.pop();
  if (regst_q.empty()) { readable_regst_desc_cnt_ -= 1; }
  regst_desc_id2is_processed_.at(cur_processed_regst_desc_id_) = true;
  processed_regst_desc_id_cnt_ += 1;
  cur_processed_regst_desc_id_ = -1;
  if (processed_regst_desc_id_cnt_ == regst_desc_id2is_processed_.size()) {
    AsyncSendRegstMsgToConsumer([&](Regst* regst) {
      regst->set_piece_id(cur_regst->piece_id());
      return true;
    });
    for (auto& pair : regst_desc_id2is_processed_) {
      CHECK(pair.second);
      pair.second = false;
    }
    processed_regst_desc_id_cnt_ = 0;
  }
  AsyncSendRegstMsgToProducer(cur_regst);
}

void InputWiseCompActor::AsyncReturnAllCustomizedReadableRegst() {
  CHECK_EQ(-1, cur_processed_regst_desc_id_);
  CHECK_EQ(0, processed_regst_desc_id_cnt_);
  CHECK_EQ(0, readable_regst_desc_cnt_);
}

REGISTER_ACTOR(TaskType::kReduceGlobalAdd, InputWiseCompActor);

}  // namespace oneflow
