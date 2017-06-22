#include "oneflow/core/actor/copy_actor.h"

namespace oneflow {

void CopyActor::Init(const TaskProto& task_proto, const ThreadCtx& thread_ctx) {
  Actor::Init(task_proto, thread_ctx);
}

int CopyActor::ProcessMsg(const ActorMsg& msg) {
  if (TryUpdtStateAsProducedRegst(msg.regst_warpper()->regst_raw_ptr()) != 0) {
    waiting_in_regst_.push(std::move(msg.regst_warpper()));
  }
  if (!waiting_in_regst_.empty() && IsWriteReady()) {
    uint64_t piece_id = expected_piece_id();
    std::shared_ptr<RegstWarpper> regst_wp = waiting_in_regst_.front();
    CHECK_EQ(regst_wp->piece_id(), piece_id);
    AsyncWardKernel(GenDefaultKernelCtx(),
        [this](uint64_t regst_desc_id) -> std::shared_ptr<RegstWarpper> {
      Regst* regst = GetCurWriteableRegst(regst_desc_id);
      if (regst == nullptr) {
        CHECK_EQ(regst_desc_id, waiting_in_regst_.front()->regst_desc_id());
        return waiting_in_regst_.front();
      } else {
        return std::make_shared<LocalRegstWarpper> (regst);
      }
    });
    ForEachCurWriteableRegst([&regst_wp](Regst* regst) {
      regst->set_piece_id(regst_wp->piece_id());
      regst->set_model_version_id(regst_wp->model_version_id());
    });
    AsyncSendReadableRegstMsg();
    AsyncSendRegstMsgToProducer(waiting_in_regst_.front());
    waiting_in_regst_.pop();
  }
  return 1;
}

}  // namespace oneflow
