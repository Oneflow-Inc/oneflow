#include "oneflow/core/actor/copy_actor.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/register/local_register_warpper.h"

namespace oneflow {

void CopyActor::Init(const TaskProto& task_proto) {
  Actor::Init(task_proto);
}

void CopyActor::ProcessMsgWithKernelCtx(const ActorMsg& msg,
                                        const KernelContext& kernel_ctx) {
  if (TryUpdtStateAsFromRegstReader(msg.regst_warpper()->regst_raw_ptr()) != 0) {
    waiting_in_regst_.push(std::move(msg.regst_warpper()));
  }
  if (!waiting_in_regst_.empty() && IsWriteReady()) {
    uint64_t piece_id = expected_piece_id();
    std::shared_ptr<RegstWarpper> regst_wp = waiting_in_regst_.front();
    CHECK_EQ(regst_wp->piece_id(), piece_id);
    WardKernel(kernel_ctx, [this](uint64_t regst_desc_id) -> std::shared_ptr<RegstWarpper> {
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
    CurWriteDone();
    ActorMsgBus::Singleton().SendMsg(ActorMsg::BuildMsgForRegstWriter(
          regst_wp->producer_actor_id(),
          regst_wp->regst_raw_ptr()));
    waiting_in_regst_.pop();
  }
}

}  // namespace oneflow
