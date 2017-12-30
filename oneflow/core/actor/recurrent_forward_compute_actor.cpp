#include "oneflow/core/actor/recurrent_forward_compute_actor.h"

namespace oneflow {

void RecurrentForwardCompActor::VirtualCompActorInit(
    const TaskProto& task_proto) {
  in_regst_desc_id_ = RegstDescId4Name("in");
  initial_hidden_regst_desc_id_ = RegstDescId4Name("initial_hidden");
  model_regst_desc_id_ = RegstDescId4Name("model");
  out_regst_desc_id_ = RegstDescId4Name("out");

  is_in_eord_ = false;
  latest_model_regst_ = nullptr;
  cur_model_regst_ = nullptr;
  out_regst_ = nullptr;

  OF_SET_MSG_HANDLER(&RecurrentForwardCompActor::HandlerInitModel);
}

int RecurrentForwardCompActor::HandlerInitModel(const ActorMsg& msg) {
  Regst* model_regst = msg.regst();
  CHECK_EQ(model_regst_desc_id_, model_regst->regst_desc_id());
  for (const ExecKernel& exec_kernel : exec_kernel_vec()) {
    exec_kernel.kernel->InitModelBlobs(
        GenDefaultKernelCtx(), parallel_ctx(),
        SnapshotMgr::Singleton()->GetReadableSnapshot(),
        [&](const std::string& bn_in_op) {
          const std::string& lbn = exec_kernel.kernel->Lbn4BnInOp(bn_in_op);
          return model_regst->GetBlobByLbn(lbn);
        });
  }
  AsyncSendRegstMsgToProducer(model_regst);
  OF_SET_MSG_HANDLER(&RecurrentForwardCompActor::HandlerNormal);
  return 0;
}

int RecurrentForwardCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    if (msg.eord_regst_desc_id() == in_regst_desc_id_) { is_in_eord_ = true; }
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* cur_regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(cur_regst) != 0) {
      int64_t cur_regst_desc_id = cur_regst->regst_desc_id();
      if (cur_regst_desc_id == in_regst_desc_id_) {
        in_regsts_.push(cur_regst);
      } else if (cur_regst_desc_id == initial_hidden_regst_desc_id_) {
        initial_hidden_regsts_.push(cur_regst);
      } else if (cur_regst_desc_id == model_regst_desc_id_) {
        if (cur_model_regst_ != latest_model_regst_) {
          AsyncSendRegstMsgToProducer(latest_model_regst_);
        }
        latest_model_regst_ = cur_regst;
      } else if (cur_regst_desc_id == out_regst_desc_id_) {
        CHECK(!out_regst_);
        out_regst_ = cur_regst;
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

  Regst* in_regst = in_regsts_.front();
  if (in_regst->piece_status().col_id() == 0) {
    if (initial_hidden_regst_desc_id_ != -1 && initial_hidden_regsts_.empty()) {
      return false;
    }
    cur_model_regst_ = latest_model_regst_;
    return true;
  } else {
    if (!out_regst_) { return false; }
    CHECK(in_regst->piece_status().IsNextColOf(out_regst_->piece_status()));
    return true;
  }
}

bool RecurrentForwardCompActor::IsReadAlwaysUnReadyFromNow() {
  return is_in_eord_ && in_regsts_.empty();
}

void RecurrentForwardCompActor::Act() {
  Regst* in_regst = in_regsts_.front();
  in_regsts_.pop();

  AsyncLaunchKernel(GenDefaultKernelCtx(),
                    [&](int64_t regst_desc_id) -> Regst* { return nullptr; });
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_model_version_id(cur_model_regst_->model_version_id());
    regst->set_is_forward(true);
    return true;
  });

  if (in_regst->piece_status().IsLastCol()) {
    if (cur_model_regst_ != latest_model_regst_) {
      AsyncSendRegstMsgToProducer(cur_model_regst_);
    } else {
      int64_t last_pid =
          GetLastPieceIdForModelVersionId(cur_model_regst_->model_version_id());
      if (JobDesc::Singleton()->IsTrain()
          && in_regst->piece_status().piece_id() == last_pid) {
        AsyncSendRegstMsgToProducer(cur_model_regst_);
        latest_model_regst_ = nullptr;
      }
    }
    cur_model_regst_ = nullptr;
  }
  if (initial_hidden_regst_desc_id_ != -1
      && in_regst->piece_status().col_id() == 0) {
    AsyncSendRegstMsgToProducer(initial_hidden_regsts_.front());
    initial_hidden_regsts_.pop();
  }
  if (out_regst_) {
    AsyncSendRegstMsgToProducer(out_regst_);
    out_regst_ = nullptr;
  }
  AsyncSendRegstMsgToProducer(in_regst);
}

void RecurrentForwardCompActor::AsyncReturnAllReadableRegst() {
  CHECK(in_regsts_.empty());
  CHECK(initial_hidden_regsts_.empty());
  CHECK(!cur_model_regst_);
  CHECK(!out_regst_);
  if (latest_model_regst_) {
    AsyncSendRegstMsgToProducer(latest_model_regst_);
    latest_model_regst_ = nullptr;
  }
}

REGISTER_ACTOR(TaskType::kRecurrentForward, RecurrentForwardCompActor);

}  // namespace oneflow
