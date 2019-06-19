#include "oneflow/core/actor/reentrant_lock_compute_actor.h"

namespace oneflow {

void ReentrantLockCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  CHECK_EQ(1, exec_kernel_vec().size());
  const auto& kernel_conf = task_proto.exec_sequence().exec_node().Get(0).kernel_conf();
  const auto& ibns = kernel_conf.op_attribute().input_bns();
  for (const auto& ibn : ibns) {
    CHECK(regst_desc_id2ibn_.emplace(exec_kernel_vec().at(0).bn_in_op2regst_desc_id.at(ibn), ibn)
              .second);
  }
  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    for (const int64_t regst_desc_id : pair.second.regst_desc_id()) {
      consumed_rs_.InsertRegstDescId(regst_desc_id);
    }
  }
  consumed_rs_.InitedDone();
  cur_processed_regst_desc_id_ = -1;
  reentrant_lock_status_.Init(kernel_conf);
  OF_SET_MSG_HANDLER(&ReentrantLockCompActor::HandlerNormal);
}

void ReentrantLockCompActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  CHECK_EQ(0, consumed_rs_.TryPushBackRegst(msg.regst()));
}

bool ReentrantLockCompActor::IsCustomizedReadReady() { return -1 != GetCurProcessedRegstDescId(); }

void ReentrantLockCompActor::ForEachCurCustomizedReadableRegst(
    std::function<void(const Regst*)> handler) const {
  handler(consumed_rs_.Front(cur_processed_regst_desc_id_));
}

void ReentrantLockCompActor::Act() {
  cur_processed_regst_desc_id_ = GetCurProcessedRegstDescId();
  Regst* cur_regst = consumed_rs_.Front(cur_processed_regst_desc_id_);
  CHECK(cur_regst);
  reentrant_lock_status_.set_cur_ibn(Ibn4RegstDescId(cur_processed_regst_desc_id_));
  reentrant_lock_status_.set_cur_act_id(act_id());
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = &reentrant_lock_status_;
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    if (cur_processed_regst_desc_id_ != regst_desc_id) { return nullptr; }
    return cur_regst;
  });
}

bool ReentrantLockCompActor::IsCustomizedReadAlwaysUnReadyFromNow() {
  return ReceiveAllEordMsg() && reentrant_lock_status_.queued_request_lock_num() == 0
         && reentrant_lock_status_.acquired_lock_num() == 0;
}

void ReentrantLockCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  if (reentrant_lock_status_.cur_act_one_lock_acquired() == false) { return; }
  HandleProducedNaiveDataRegstToConsumer([this](Regst* regst) { return true; });
}

void ReentrantLockCompActor::AsyncSendCustomizedConsumedRegstMsgToProducer() {
  Regst* cur_regst = consumed_rs_.Front(cur_processed_regst_desc_id_);
  CHECK(cur_regst);
  AsyncSendRegstMsgToProducer(cur_regst);
  CHECK_EQ(0, consumed_rs_.TryPopFrontRegst(cur_processed_regst_desc_id_));
  cur_processed_regst_desc_id_ = -1;
}

void ReentrantLockCompActor::AsyncReturnAllCustomizedReadableRegst() {
  CHECK_EQ(-1, cur_processed_regst_desc_id_);
  CHECK_EQ(0, consumed_rs_.available_regst_desc_cnt());
}

int64_t ReentrantLockCompActor::GetCurProcessedRegstDescId() {
  int64_t cur_processed_regst_desc_id = -1;
  consumed_rs_.ForChosenRegstDeq(
      [&cur_processed_regst_desc_id](int64_t) { return cur_processed_regst_desc_id == -1; },
      [&cur_processed_regst_desc_id](const std::deque<Regst*>& reg_deq) {
        if (reg_deq.empty()) { return; }
        cur_processed_regst_desc_id = reg_deq.front()->regst_desc_id();
      });
  return cur_processed_regst_desc_id;
}

REGISTER_ACTOR(kReentrantLock, ReentrantLockCompActor);

}  // namespace oneflow
