#include "oneflow/core/actor/forward_compute_actor.h"

namespace oneflow {

void ForwardCompActor::VirtualCompActorInit(const TaskProto& task_proto,
                                            const ThreadCtx& thread_ctx) {
  in_desc_id_ = RegstDescId4Name("in");
  CHECK_NE(in_desc_id_, -1);
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  model_regst_ = nullptr;
  model_tmp_regst_ = nullptr;
  set_num_of_remaining_eord(1 + (model_regst_desc_id_ != -1)
                            + (model_tmp_regst_desc_id_ != -1));
  mut_num_of_read_empty() = 1;  // only consider "in"regst
  OF_SET_MSG_HANDLER(&ForwardCompActor::HandlerNormal);
}

bool ForwardCompActor::IsReadReady() {
  if (in_desc_id_ == -1) { return true; }
  if (in_desc_id_ == -2) { return false; }
  if (in_.empty() || (model_regst_desc_id_ != -1 && !model_regst_)
      || (model_tmp_regst_desc_id_ != -1 && !model_tmp_regst_)) {
    return false;
  }
  if (JobDesc::Singleton()->IsTrain() && model_regst_desc_id_ != -1) {
    // Ho Q, Cipar J, Cui H, et al. More effective distributed ml via a stale
    // synchronous parallel parameter server
    int32_t staleness = JobDesc::Singleton()->Staleness();
    int32_t num_of_pieces_in_batch = JobDesc::Singleton()->NumOfPiecesInBatch();
    int64_t cur_iteration = in_.front()->piece_id() / num_of_pieces_in_batch;
    int64_t stale_version = cur_iteration - staleness;
    return model_regst_->model_version_id() >= stale_version;
  }
  return true;
}

void ForwardCompActor::AsyncSendMsgToModelAndModelTmpProducer() {
  if (model_regst_desc_id_ != -1) {
    AsyncSendRegstMsgToProducer(model_regst_);
    model_regst_ = nullptr;
  }
  if (model_tmp_regst_desc_id_ != -1) {
    AsyncSendRegstMsgToProducer(model_tmp_regst_);
    model_tmp_regst_ = nullptr;
  }
}

int ForwardCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kEORD);
    ProcessOneEord();
    if (msg_handler() == &ForwardCompActor::HandlerZombie
        || msg_handler() == nullptr) {
      AsyncSendMsgToModelAndModelTmpProducer();
    }
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(regst) != 0) {
      if (regst->regst_desc_id() == model_tmp_regst_desc_id_) {
        CHECK(!model_tmp_regst_);
        model_tmp_regst_ = regst;
        readable_regst_[model_tmp_regst_desc_id_] = regst;
      } else if (regst->regst_desc_id() == model_regst_desc_id_) {
        if (model_regst_) { AsyncSendRegstMsgToProducer(model_regst_); }
        model_regst_ = regst;
        readable_regst_[model_regst_desc_id_] = regst;
      } else {
        in_.push(regst);
        mut_num_of_read_empty() = 0;
      }
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return msg_handler() == nullptr;
}

int ForwardCompActor::HandlerWaitUntilNoReadableRegst(const ActorMsg& msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst()), 0);
  ActUntilFail();
  CHECK_NE(in_desc_id_, -1);
  if (in_.empty()) {
    AsyncSendMsgToModelAndModelTmpProducer();
    AsyncSendEORDMsgForAllProducedRegstDesc();
    OF_SET_MSG_HANDLER(&ForwardCompActor::HandlerZombie);
  }
  return 0;
}

void ForwardCompActor::Act() {
  int64_t piece_id = 0;  // expected_piece_id();
  if (!in_.empty()) {
    CHECK_EQ(in_.front()->piece_id(), piece_id);
    readable_regst_[in_.front()->regst_desc_id()] = in_.front();
  }
  int64_t model_version_id = -1;
  if (model_regst_) { model_version_id = model_regst_->model_version_id(); }
  AsyncLaunchKernel(GenDefaultKernelCtx(),
                    [this](int64_t regst_desc_id) -> Regst* {
                      Regst* regst = GetCurWriteableRegst(regst_desc_id);
                      if (regst == nullptr) {
                        return readable_regst_.at(regst_desc_id);
                      } else {
                        return regst;
                      }
                    });
  AsyncSendRegstMsgToConsumer([piece_id, model_version_id](Regst* regst) {
    regst->set_piece_id(piece_id);
    regst->set_model_version_id(model_version_id);
  });
  if (!in_.empty()) {
    AsyncSendRegstMsgToProducer(in_.front());
    in_.pop();
    mut_num_of_read_empty() = in_.empty();
  }
  TODO();
  // if (expected_piece_id() == JobDesc::Singleton()->total_piece_num()) {
  //  in_desc_id_ = -2;
  //  AsyncSendMsgToModelAndModelTmpProducer();
  //  AsyncSendEORDMsgForAllProducedRegstDesc();
  //  TrySwitchToZombie();
  //}
}

}  // namespace oneflow
