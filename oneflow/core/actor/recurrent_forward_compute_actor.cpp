#include "oneflow/core/actor/recurrent_forward_compute_actor.h"

namespace oneflow {

void RecurrentForwardCompActor::VirtualForwardCompActorInit(
    const TaskProto& task_proto) {
  initial_hidden_regst_desc_id_ = RegstDescId4Name("initial_hidden");
  out_regst_desc_id_ = RegstDescId4Name("out");

  latest_model_regst_ = nullptr;
  cur_model_regst_ = nullptr;
  out_regst_ = nullptr;

  OF_SET_MSG_HANDLER(&RecurrentForwardCompActor::HandlerInitModel);
}

void RecurrentForwardCompActor::SetMsgHandlerOfNormal() {
  OF_SET_MSG_HANDLER(&RecurrentForwardCompActor::HandlerNormal);
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
      } else if (cur_regst_desc_id == initial_hidden_regst_desc_id_) {
        initial_hidden_regsts_.push(cur_regst);
      } else if (cur_regst_desc_id == model_regst_desc_id()) {
        if (cur_model_regst_ != latest_model_regst_) {
          AsyncSendRegstMsgToProducer(latest_model_regst_);
        } else {
          latest_model_regst_ = cur_regst;
        }
      } else if (cur_regst_desc_id == out_regst_desc_id_) {
        CHECK(!out_regst_);
        if (cur_regst->IsLastCol()) {
          AsyncSendRegstMsgToProducer(cur_regst);
        } else {
          out_regst_ = cur_regst;
        }
      } else if (cur_regst_desc_id == model_tmp_regst_desc_id()) {
        CHECK(!model_tmp_regst());
        set_mode_tmp_regst(cur_regst);
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
    if (initial_hidden_regst_desc_id_ != -1 && initial_hidden_regsts_.empty()) {
      return false;
    }
    cur_model_regst_ = latest_model_regst_;
    return true;
  } else {
    if (!out_regst_) { return false; }
    CHECK(in_regst->HaveNextPieceColStatusOf(out_regst_));
    return true;
  }
}

bool RecurrentForwardCompActor::IsReadAlwaysUnReadyFromNow() {
  return is_in_eord() && in_regsts_.empty();
}

void RecurrentForwardCompActor::Act() {
  Regst* in_regst = in_regsts_.front();
  in_regsts_.pop();

  AsyncLaunchRecurrentKernel(
      GenDefaultKernelCtx(),
      [&](int64_t regst_desc_id, const std::string& bn_in_op) -> Regst* {
        if (regst_desc_id == in_regst_desc_id()) {
          return in_regst;
        } else if (regst_desc_id == model_regst_desc_id()) {
          return cur_model_regst_;
        } else if (regst_desc_id == model_tmp_regst_desc_id()) {
          return model_tmp_regst();
        } else if (regst_desc_id == initial_hidden_regst_desc_id_) {
          return initial_hidden_regsts_.front();
        } else if (regst_desc_id == out_regst_desc_id_ && bn_in_op == "ht_1") {
          return out_regst_;
        } else {
          return GetCurWriteableRegst(regst_desc_id);
        }
      });
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_piece_id(in_regst->piece_id());
    regst->set_model_version_id(cur_model_regst_->model_version_id());
    return true;
  });

  if (in_regst->IsLastCol()) {
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
  if (initial_hidden_regst_desc_id_ != -1 && in_regst->col_id() == 0) {
    AsyncSendRegstMsgToProducer(initial_hidden_regsts_.front());
    initial_hidden_regsts_.pop();
  }
  if (out_regst_) {
    AsyncSendRegstMsgToProducer(out_regst_);
    out_regst_ = nullptr;
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
  CHECK(initial_hidden_regsts_.empty());
  CHECK(!cur_model_regst_);
  CHECK(!out_regst_);
}

REGISTER_ACTOR(TaskType::kRecurrentForward, RecurrentForwardCompActor);

}  // namespace oneflow
