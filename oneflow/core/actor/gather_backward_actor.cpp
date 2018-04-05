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
    if (msg.eord_regst_desc_id() == out_diff_regst_desc_id_) {
      is_out_diff_eord_ = true;
    }
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* cur_regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(cur_regst) != 0) {
      if (cur_regst->regst_desc_id() == in_regst_desc_id_) {
        if (cur_regst->IsMaxCol()) {
          max_col_in_regst_of_pieces_.push(cur_regst);
        } else {
          AsyncSendRegstMsgToProducer(cur_regst);
        }
      } else {
        CHECK_EQ(out_diff_regst_desc_id_, cur_regst->regst_desc_id());
        out_diff_regsts_.push(cur_regst);
      }
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
}

bool GatherBackwardActor::IsReadReady() {
  return out_diff_regsts_.empty() == false
         && max_col_in_regst_of_pieces_.empty() == false;
}

void GatherBackwardActor::Act() {
  Regst* cur_out_diff_regst = out_diff_regsts_.front();
  Regst* cur_in_regst = max_col_in_regst_of_pieces_.front();
  if (cur_generated_cid_ == -1) {
    cur_generated_cid_ = cur_in_regst->max_col_id();
  }
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = &cur_generated_cid_;
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    if (regst_desc_id == out_diff_regst_desc_id_) {
      return cur_out_diff_regst;
    } else if (regst_desc_id == in_regst_desc_id_) {
      return cur_in_regst;
    } else {
      return GetCurWriteableRegst(regst_desc_id);
    }
  });
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_piece_id(cur_in_regst->piece_id());
    regst->set_col_id(cur_generated_cid_);
    regst->set_max_col_id(cur_in_regst->max_col_id());
    return true;
  });
  cur_generated_cid_ -= 1;
  if (cur_generated_cid_ == -1) {
    max_col_in_regst_of_pieces_.pop();
    out_diff_regsts_.pop();
    AsyncSendRegstMsgToProducer(cur_in_regst);
    AsyncSendRegstMsgToProducer(cur_out_diff_regst);
  }
}

bool GatherBackwardActor::IsReadAlwaysUnReadyFromNow() {
  return is_out_diff_eord_ && out_diff_regsts_.empty();
}

void GatherBackwardActor::AsyncReturnAllReadableRegst() {
  CHECK(out_diff_regsts_.empty());
  CHECK(max_col_in_regst_of_pieces_.empty());
}

void GatherBackwardActor::ForEachCurReadableRegst(
    std::function<void(const Regst*)> handler) {
  handler(out_diff_regsts_.front());
}

REGISTER_ACTOR(TaskType::kGatherBackward, GatherBackwardActor);

}  // namespace oneflow
