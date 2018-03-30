#include "oneflow/core/actor/gather_backward_actor.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

void GatherBackwardActor::VirtualActorInit(const TaskProto& task_proto) {
  in_regst_desc_id_ = RegstDescId4Name("in");
  out_diff_regst_desc_id_ = RegstDescId4Name("out_diff");
  is_out_diff_eord_ = false;
  cur_generated_cid_ = -1;
  OF_SET_MSG_HANDLER(&GatherBackwardActor::HandlerNormal);
}

int GatherBackwardActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    DecreaseRemainingEordCnt();
    is_out_diff_eord_ = true;
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* cur_regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(cur_regst) != 0) {
      if (cur_regst->regst_desc_id() == in_regst_desc_id_) {
        AsyncSendRegstMsgToProducer(cur_regst);
      } else {
        CHECK_EQ(out_diff_regst_desc_id_, cur_regst->regst_desc_id());
        out_diff_regst_.push(cur_regst);
      }
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
}

void GatherBackwardActor::Act() {
  Regst* cur_regst = out_diff_regst_.front();
  if (cur_generated_cid_ == -1) {
    cur_generated_cid_ = cur_regst->max_col_id();
  }
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = &cur_generated_cid_;
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    if (regst_desc_id == cur_regst->regst_desc_id()) {
      return cur_regst;
    } else {
      return GetCurWriteableRegst(regst_desc_id);
    }
  });
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_piece_id(cur_regst->piece_id());
    regst->set_col_id(cur_generated_cid_);
    regst->set_max_col_id(cur_regst->max_col_id());
    return true;
  });
  cur_generated_cid_ -= 1;
  if (cur_generated_cid_ == -1) { out_diff_regst_.pop(); }
}

bool GatherBackwardActor::IsReadAlwaysUnReadyFromNow() {
  return is_out_diff_eord_ && out_diff_regst_.empty();
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
