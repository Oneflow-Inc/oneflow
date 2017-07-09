#include "oneflow/core/actor/boxing_actor.h"
#include "oneflow/core/actor/actor_registry.h"
#include "oneflow/core/register/local_register_wrapper.h"

namespace oneflow {

void BoxingActor::Init(const TaskProto& task_proto,
                       const ThreadCtx& thread_ctx) {
  Actor::Init(task_proto, thread_ctx);
  int num_of_subscribed_regsts = task_proto.subscribed_regst_desc_id().size();
  mut_num_of_not_eord() = num_of_subscribed_regsts;
  mut_num_of_read_empty() = num_of_subscribed_regsts;
  CHECK(thread_ctx.cpu_stream);
  mut_device_ctx().reset(new CpuDeviceCtx(thread_ctx.cpu_stream));
  OF_SET_MSG_HANDLE(&BoxingActor::HandleNormal);
}

int BoxingActor::HandleNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kEORD);
    ProcessEord();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst_wrapper()->regst_raw_ptr())
        != 0) {
      std::shared_ptr<RegstWrapper> regst_wp = msg.regst_wrapper();
      mut_num_of_read_empty() -= read_regst_[regst_wp->regst_desc_id()].empty();
      read_regst_.at(regst_wp->regst_desc_id()).push(regst_wp);
    } else {
      // do nothing
    }
    ActUntilFail();
  }
  return msg_handle() == nullptr;
}

int BoxingActor::HandleWaitUntilNoReadableRegst(const ActorMsg& msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst_wrapper()->regst_raw_ptr()),
           0);
  ActUntilFail();
  if (mut_num_of_read_empty()) {
    AsyncSendEORDMsgForAllProducedRegstDesc();
    OF_SET_MSG_HANDLE(&BoxingActor::HandleWaitUntilReadingCntEqualZero);
  }
  return 0;
}

void BoxingActor::Act() {
  int64_t piece_id = expected_piece_id();
  for (const auto& pair : read_regst_) {
    CHECK_EQ(pair.second.front()->piece_id(), piece_id);
  }
  AsyncLaunchKernel(
      GenDefaultKernelCtx(),
      [this](int64_t regst_desc_id) -> std::shared_ptr<RegstWrapper> {
        Regst* regst = GetCurWriteableRegst(regst_desc_id);
        if (regst == nullptr) {
          return read_regst_.at(regst_desc_id).front();
        } else {
          return std::make_shared<LocalRegstWrapper>(regst);
        }
      });
  AsyncSendReadableRegstMsg(
      [piece_id](Regst* regst) { regst->set_piece_id(piece_id); });
  for (auto& pair : read_regst_) {
    AsyncSendRegstMsgToProducer(pair.second.front());
    pair.second.pop();
    mut_num_of_read_empty() += pair.second.empty();
  }
}

REGISTER_ACTOR(kBoxingTask, true, BoxingActor);
REGISTER_ACTOR(kBoxingTask, false, BoxingActor);

}  // namespace oneflow
