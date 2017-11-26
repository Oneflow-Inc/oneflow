#include "oneflow/core/actor/boxing_actor.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

void BoxingActor::VirtualActorInit(const TaskProto& task_proto) {
  is_eord_ = false;
  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    readable_regst_[pair.second] = {};
  }
  readable_regst_cnt_ = 0;
  OF_SET_MSG_HANDLER(&BoxingActor::HandlerNormal);
}

int BoxingActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    is_eord_ = true;
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst()) != 0) {
      std::queue<Regst*>& rq = readable_regst_.at(msg.regst()->regst_desc_id());
      if (rq.empty()) { readable_regst_cnt_ += 1; }
      rq.push(msg.regst());
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
}

void BoxingActor::Act() {
  int64_t piece_id = readable_regst_.begin()->second.front()->piece_id();
  AsyncLaunchKernel(GenDefaultKernelCtx(),
                    [this](int64_t regst_desc_id) -> Regst* {
                      Regst* regst = GetCurWriteableRegst(regst_desc_id);
                      if (regst == nullptr) {
                        return readable_regst_.at(regst_desc_id).front();
                      } else {
                        return regst;
                      }
                    });
  AsyncSendRegstMsgToConsumer(
      [&](Regst* regst) { regst->set_piece_id(piece_id); });
  for (auto& pair : readable_regst_) {
    AsyncSendRegstMsgToProducer(pair.second.front());
    pair.second.pop();
    if (pair.second.empty()) { readable_regst_cnt_ -= 1; }
  }
}

bool BoxingActor::IsReadReady() {
  return readable_regst_.size() == readable_regst_cnt_;
}

bool BoxingActor::IsReadAlwaysUnReadyFromNow() {
  return is_eord_ && readable_regst_cnt_ == 0;
}

void BoxingActor::AsyncReturnAllReadableRegst() {
  CHECK_EQ(readable_regst_cnt_, 0);
}

REGISTER_ACTOR(TaskType::kBoxing, BoxingActor);

}  // namespace oneflow
