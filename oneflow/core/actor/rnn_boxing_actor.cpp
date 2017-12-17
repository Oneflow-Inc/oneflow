#include "oneflow/core/actor/boxing_actor.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

void BoxingActor::VirtualActorInit(const TaskProto& task_proto) {
  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    is_finished_in_cur_pid_[pair.second] = false;
  }
  OF_SET_MSG_HANDLER(&BoxingActor::HandlerNormal);
}

int BoxingActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst()) != 0) {
      int64_t cur_pid = msg.regst()->piece_status().piece_id();
      readable_regst_[cur_pid][msg.regst()->regst_desc_id()].push(msg.regst());
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
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_piece_id(piece_id);
    return true;
  });
  for (auto& pair : readable_regst_) {
    AsyncSendRegstMsgToProducer(pair.second.front());
    pair.second.pop();
    if (pair.second.empty()) { readable_regst_cnt_ -= 1; }
  }
}

bool BoxingActor::IsReadReady() {
  const auto& cur_readable_regst = readable_regst_.begin()->second;
  for (const auto& pair: is_finished_in_cur_pid_) {
    if (cur_readable_regst.find(pair.first) == cur_readable_regst.end() &&
        !pair.second) {
      return false;
    }
  }
  return true;
}

bool BoxingActor::IsReadAlwaysUnReadyFromNow() {
  return !remaining_eord_cnt() && readable_regst_.empty();
}

void BoxingActor::AsyncReturnAllReadableRegst() {
  CHECK(readable_regst_.empty());
}

REGISTER_ACTOR(TaskType::kBoxing, BoxingActor);

}  // namespace oneflow
