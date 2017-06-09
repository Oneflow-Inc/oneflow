#include "oneflow/core/actor/copy_comm_net_actor.h"
#include "oneflow/core/actor/actor_registry.h"
#include "oneflow/core/register/local_register_warpper.h"

namespace oneflow {

void CopyCommNetActor::Init(const TaskProto& task_proto) {
  Actor::Init(task_proto);
}

void CopyCommNetActor::ProcessMsg(const ActorMsg& msg,
                                  const ThreadContext&) {
  KernelContext kernel_ctx;
  if (TryUpdtStateAsFromRegstReader(msg.regst_warpper()->regst_raw_ptr()) != 0) {
    waiting_in_regst_.push(std::move(msg.regst_warpper()));
  }
  if (!waiting_in_regst_.empty() && IsWriteReady()) {
    WardKernel(kernel_ctx, [this](uint64_t regst_desc_id) -> std::shared_ptr<RegstWarpper> {
      Regst* regst = GetCurWriteableRegst(regst_desc_id);
      if (regst == nullptr) {
        CHECK_EQ(regst_desc_id, waiting_in_regst_.front()->regst_desc_id());
        return waiting_in_regst_.front();
      } else {
        return std::make_shared<LocalRegstWarpper> (regst);
      }
    });
    CurWriteDone();
    std::shared_ptr<RegstWarpper> regst = waiting_in_regst_.front();
    ActorMsgBus::Singleton().SendMsg(ActorMsg::BuildMsgForRegstWriter(
          regst->producer_actor_id(),
          regst->regst_raw_ptr()));
    waiting_in_regst_.pop();
  }
}

REGISTER_ACTOR(kCopyCommNetTask, true, CopyCommNetActor);
REGISTER_ACTOR(kCopyCommNetTask, false, CopyCommNetActor);

}  // namespace oneflow
