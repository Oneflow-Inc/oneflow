#include "oneflow/core/actor/copy_actor.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/register/local_register_warpper.h"

namespace oneflow {

void CopyActor::Init(const TaskProto& task_proto) {
  Actor::Init(task_proto);
}

void CopyActor::ProcessMsgWithKernelCtx(const ActorMsg& msg,
                                        const KernelContext& kernel_ctx) {
  if (TryOneReadDone(msg.regst_warpper()->regst_raw_ptr()) != 0) {
    waiting_in_regst_.push(std::move(msg.regst_warpper()));
  }
  if (!waiting_in_regst_.empty() && IsWriteReady()) {
    uint64_t piece_id = expected_piece_id();
    CHECK_EQ(waiting_in_regst_.front()->piece_id(), piece_id);
    WardKernel(kernel_ctx, [this](uint64_t regst_desc_id) -> std::shared_ptr<RegstWarpper> {
      Regst* regst = GetCurWriteableRegst(regst_desc_id);
      if (regst == nullptr) {
        CHECK_EQ(regst_desc_id, waiting_in_regst_.front()->regst_desc_id());
        return waiting_in_regst_.front();
      } else {
        return std::make_shared<LocalRegstWarpper> (regst);
      }
    });
    ForEachCurWriteableRegst([piece_id](Regst* regst) {
      regst->set_piece_id(piece_id);
    });
    CurWriteDone();
    std::shared_ptr<RegstWarpper> regst = waiting_in_regst_.front();
    ActorMsgBus::Singleton().SendMsg(ActorMsg::BuildMsgForRegstWriter(
          regst->producer_actor_id(),
          regst->regst_raw_ptr()));
    waiting_in_regst_.pop();
  }
}

}  // namespace oneflow
