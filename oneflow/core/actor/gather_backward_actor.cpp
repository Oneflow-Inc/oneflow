#include "oneflow/core/actor/gather_backward_actor.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

void GatherBackwardActor::VirtualActorInit(const TaskProto& task_proto) {
  is_in_eord_ = false;
  cur_generating_cid_ = -1;
  OF_SET_MSG_HANDLER(&GatherBackwardActor::HandlerNormal);
}

int GatherBackwardActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    DecreaseRemainingEordCnt();
    is_in_eord_ = true;
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst()) != 0) {
      out_diff_regst_.push(msg.regst());
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
}

void GatherBackwardActor::Act() {
  Regst* in_regst = out_diff_regst_.front();
  if (cur_generating_cid_ == -1) cur_generating_cid_ = in_regst->max_col_id();
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = &cur_generating_cid_;
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    if (regst_desc_id == in_regst->regst_desc_id()) {
      return in_regst;
    } else {
      return GetCurWriteableRegst(regst_desc_id);
    }
  });
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_piece_id(in_regst->piece_id());
    regst->set_col_id(cur_generating_cid_);
    regst->set_max_col_id(in_regst->max_col_id());
    return true;
  });
  cur_generating_cid_ -= 1;
}

bool GatherBackwardActor::IsReadAlwaysUnReadyFromNow() {
  return is_in_eord_ && out_diff_regst_.empty();
}

void GatherBackwardActor::AsyncReturnAllReadableRegst() {
  CHECK(out_diff_regst_.empty());
}

void GatherBackwardActor::ForEachCurReadableRegst(
    std::function<void(const Regst*)> handler) {
  handler(out_diff_regst_.front());
}

REGISTER_ACTOR(TaskType::kGatherBackward, GatherBackwardActor);

}  // namespace oneflow
