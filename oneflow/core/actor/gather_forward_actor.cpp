#include "oneflow/core/actor/gather_forward_actor.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

void GatherForwardActor::VirtualActorInit(const TaskProto& task_proto) {
  is_in_eord_ = false;
  OF_SET_MSG_HANDLER(&GatherForwardActor::HandlerNormal);
}

int GatherForwardActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    is_in_eord_ = true;
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst()) != 0) {
      pending_in_regst_.push(msg.regst());
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
}

void GatherForwardActor::Act() {
  Regst* in_regst = pending_in_regst_.front();
  pending_in_regst_.pop();
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  int32_t col_id = in_regst->col_id();
  kernel_ctx.other = &col_id;
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    if (regst_desc_id == in_regst->regst_desc_id()) {
      return in_regst;
    } else {
      return GetCurWriteableRegst(regst_desc_id);
    }
  });
  if (in_regst->col_id() == in_regst->max_col_id()) {
    AsyncSendRegstMsgToConsumer([&](Regst* regst) {
      regst->set_piece_id(in_regst->piece_id());
      regst->set_col_id(0);
      regst->set_max_col_id(0);
      return true;
    });
  }
  AsyncSendRegstMsgToProducer(in_regst);
}

bool GatherForwardActor::IsReadAlwaysUnReadyFromNow() {
  return is_in_eord_ && pending_in_regst_.empty();
}

void GatherForwardActor::AsyncReturnAllReadableRegst() {
  CHECK(pending_in_regst_.empty());
}

void GatherForwardActor::ForEachCurReadableRegst(
    std::function<void(const Regst*)> handler) {
  handler(pending_in_regst_.front());
}

REGISTER_ACTOR(TaskType::kGatherForward, GatherForwardActor);

}  // namespace oneflow
