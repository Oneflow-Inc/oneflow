#include "oneflow/core/actor/recurrent_forward_compute_actor.h"

namespace oneflow {

void RecurrentForwardCompActor::VirtualForwardCompActorInit(
    const TaskProto& task_proto) {
  h0_regst_desc_id_ = RegstDescId4Name("h0");
  rec_in_regst_desc_id_ = RegstDescId4Name("rec_in");
  if (rec_in_regst_desc_id_ == -1) {
    CHECK(parallel_ctx()->policy() == kDataParallel);
  } else {
    CHECK(parallel_ctx()->policy() == kModelParallel);
  }

  latest_model_regst_ = nullptr;
  cur_model_regst_ = nullptr;
  rec_in_regst_ = nullptr;

  OF_SET_MSG_HANDLER(&RecurrentForwardCompActor::HandlerInitModel);
}

int RecurrentForwardCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    if (msg.eord_regst_desc_id() == in_regst_desc_id()) {
      set_is_in_eord(true);
    }
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* cur_regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(cur_regst) != 0) {
      int64_t cur_regst_desc_id = cur_regst->regst_desc_id();
      if (cur_regst_desc_id == in_regst_desc_id()) {
        in_regsts_.push(cur_regst);
      } else if (cur_regst_desc_id == h0_regst_desc_id_) {
        h0_regsts_.push(cur_regst);
      } else if (cur_regst_desc_id == model_regst_desc_id()) {
        if (cur_model_regst_ != latest_model_regst_) {
          AsyncSendRegstMsgToProducer(latest_model_regst_);
        } else {
          latest_model_regst_ = cur_regst;
        }
      } else if (cur_regst_desc_id == rec_in_regst_desc_id_) {
        CHECK(!rec_in_regst_);
        if (cur_regst->IsMaxCol()) {
          AsyncSendRegstMsgToProducer(cur_regst);
        } else {
          rec_in_regst_ = cur_regst;
        }
      } else if (cur_regst_desc_id == model_tmp_regst_desc_id()) {
        CHECK(!model_tmp_regst());
        set_model_tmp_regst(cur_regst);
      } else {
        UNEXPECTED_RUN();
      }
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
}

bool RecurrentForwardCompActor::IsReadReady() {
  if (in_regsts_.empty()) { return false; }
  if (!latest_model_regst_) { return false; }
  if (model_tmp_regst_desc_id() != -1 && !model_tmp_regst()) { return false; }

  Regst* in_regst = in_regsts_.front();
  if (in_regst->col_id() == 0) {
    if (h0_regst_desc_id_ != -1 && h0_regsts_.empty()) { return false; }
    cur_model_regst_ = latest_model_regst_;
    return true;
  } else {
    if (rec_in_regst_desc_id_ != -1 && !rec_in_regst_) { return false; }
    CHECK(in_regst->HaveNextPieceColStatusOf(rec_in_regst_));
    return true;
  }
}

bool RecurrentForwardCompActor::IsReadAlwaysUnReadyFromNow() {
  return is_in_eord() && in_regsts_.empty();
}

void RecurrentForwardCompActor::Act() {
  Regst* in_regst = in_regsts_.front();
  in_regsts_.pop();

  AsyncLaunchKernel(GenDefaultKernelCtx(),
                    [&](int64_t regst_desc_id) -> Regst* {
                      if (regst_desc_id == in_regst_desc_id()) {
                        return in_regst;
                      } else if (regst_desc_id == model_regst_desc_id()) {
                        return cur_model_regst_;
                      } else if (regst_desc_id == model_tmp_regst_desc_id()) {
                        return model_tmp_regst();
                      } else if (regst_desc_id == h0_regst_desc_id_) {
                        return h0_regsts_.front();
                      } else if (regst_desc_id == rec_in_regst_desc_id_) {
                        return rec_in_regst_;
                      } else {
                        return GetCurWriteableRegst(regst_desc_id);
                      }
                    });
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_piece_id(in_regst->piece_id());
    regst->set_col_id(in_regst->col_id());
    regst->set_max_col_id(in_regst->max_col_id());
    regst->set_model_version_id(cur_model_regst_->model_version_id());
    return true;
  });

  if (in_regst->IsMaxCol()) {
    if (cur_model_regst_ != latest_model_regst_) {
      AsyncSendRegstMsgToProducer(cur_model_regst_);
    } else {
      int64_t last_pid =
          GetLastPieceIdForModelVersionId(cur_model_regst_->model_version_id());
      if (JobDesc::Singleton()->IsTrain() && in_regst->piece_id() == last_pid) {
        AsyncSendRegstMsgToProducer(cur_model_regst_);
        latest_model_regst_ = nullptr;
      }
    }
    cur_model_regst_ = nullptr;
  }
  if (h0_regst_desc_id_ != -1 && in_regst->col_id() == 0) {
    AsyncSendRegstMsgToProducer(h0_regsts_.front());
    h0_regsts_.pop();
  }
  if (rec_in_regst_) {
    AsyncSendRegstMsgToProducer(rec_in_regst_);
    rec_in_regst_ = nullptr;
  }
  AsyncSendRegstMsgToProducer(in_regst);
}

void RecurrentForwardCompActor::TryAsyncReturnModelRegst() {
  if (latest_model_regst_) {
    AsyncSendRegstMsgToProducer(latest_model_regst_);
    latest_model_regst_ = nullptr;
  }
}

void RecurrentForwardCompActor::CheckBeforeAsyncReturnAllReadableRegst() {
  CHECK(in_regsts_.empty());
  CHECK(h0_regsts_.empty());
  CHECK(!cur_model_regst_);
  CHECK(!rec_in_regst_);
}

REGISTER_ACTOR(TaskType::kRecurrentForward, RecurrentForwardCompActor);

}  // namespace oneflow
