#include "oneflow/core/actor/backward_compute_actor.h"

namespace oneflow {

void BackwardCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  activation_regst_desc_id_ = RegstDescId4Name("activation");
  data_tmp_regst_desc_id_ = RegstDescId4Name("data_tmp");
  out_regst_desc_id_ = RegstDescId4Name("out");
  OF_SET_MSG_HANDLER(&BackwardCompActor::HandlerNormal);
}

int BackwardCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    // CHECK_EQ(msg.actor_cmd(), ActorCmd::kEORD);
    ProcessOneEord();
    if (msg_handler() == &BackwardCompActor::HandlerZombie
        || msg_handler() == nullptr) {
      // AsyncSendMsgToModelAndModelTmpProducer();
    }
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(regst) != 0) {
      if (regst->regst_desc_id() == model_tmp_regst_desc_id_) {
        CHECK(readable_regsts_.find(model_tmp_regst_desc_id_)
              == readable_regsts_.end());
      } else if (regst->regst_desc_id() == model_regst_desc_id_) {
      } else {
        // do nothing
      }
      readable_regsts_.at(regst->regst_desc_id()).push(regst);
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return msg_handler() == nullptr;
}

bool BackwardCompActor::IsReadReady() {
  if (model_regst_desc_id_ != -1) {
    int cur_model_version_id =
        readable_regsts_.at(out_regst_desc_id_).front()->model_version_id();
    CHECK_GE(cur_model_version_id, 0);
    while (readable_regsts_.at(model_regst_desc_id_).front()->model_version_id()
               != cur_model_version_id
           && !readable_regsts_.at(model_regst_desc_id_).empty()) {
      AsyncSendRegstMsgToProducer(
          readable_regsts_.at(model_regst_desc_id_).front());
      readable_regsts_.at(model_regst_desc_id_).pop();
    }
  }
  return false;
}

// void BackwardCompActor::AsyncSendMsgToModelAndModelTmpProducer() {
//  while (model_regst_desc_id_ != -1
//         && !readable_regsts_.at(model_regst_desc_id_).empty()) {
//    AsyncSendRegstMsgToProducer(
//        readable_regsts_.at(model_regst_desc_id_).front());
//    readable_regsts_.at(model_regst_desc_id_).pop();
//  }
//  if (model_tmp_regst_desc_id_ != -1) {
//    AsyncSendRegstMsgToProducer(
//        readable_regsts_.at(model_tmp_regst_desc_id_).front());
//    readable_regsts_.at(model_tmp_regst_desc_id_).pop();
//  }
//}

int BackwardCompActor::HandlerUntilReadAlwaysUnReady(const ActorMsg& msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst()), 0);
  ActUntilFail();
  // AsyncSendMsgToModelAndModelTmpProducer();
  AsyncSendEORDMsgForAllProducedRegstDesc();
  OF_SET_MSG_HANDLER(&BackwardCompActor::HandlerZombie);
  return 0;
}

void BackwardCompActor::Act() {
  int64_t piece_id =
      readable_regsts_.at(out_regst_desc_id_).front()->piece_id();
  for (const auto& pair : readable_regsts_) {
    if (pair.first != model_regst_desc_id_
        && pair.first != model_tmp_regst_desc_id_) {
      CHECK_EQ(pair.second.front()->piece_id(), piece_id);
    }
  }
  AsyncLaunchKernel(GenDefaultKernelCtx(),
                    [this](int64_t regst_desc_id) -> Regst* {
                      Regst* regst = GetCurWriteableRegst(regst_desc_id);
                      if (regst == nullptr) {
                        return readable_regsts_.at(regst_desc_id).front();
                      } else {
                        return regst;
                      }
                    });
  AsyncSendRegstMsgToConsumer(
      [piece_id](Regst* regst) { regst->set_piece_id(piece_id); });
  for (auto& pair : readable_regsts_) {
    if (pair.first != model_regst_desc_id_
        && pair.first != model_tmp_regst_desc_id_) {
      AsyncSendRegstMsgToProducer(pair.second.front());
      pair.second.pop();
    }
  }
}

}  // namespace oneflow
