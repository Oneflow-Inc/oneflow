#include "oneflow/core/actor/boxing_actor.h"
#include "oneflow/core/actor/actor_registry.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

void BoxingActor::Init(const TaskProto& task_proto,
                       const ThreadCtx& thread_ctx) {
  Actor::Init(task_proto, thread_ctx);
  int num_of_consumed_regsts = task_proto.consumed_regst_desc_id().size();
  set_num_of_remaining_eord(num_of_consumed_regsts);
  mut_num_of_read_empty() = num_of_consumed_regsts;
  CHECK(thread_ctx.cpu_stream);
  mut_device_ctx().reset(new CpuDeviceCtx(thread_ctx.cpu_stream));
  OF_SET_MSG_HANDLER(&BoxingActor::HandlerNormal);
}

int BoxingActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kEORD) << actor_id();
    ProcessEord();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(regst) != 0) {
      mut_num_of_read_empty() -= read_regst_[regst->regst_desc_id()].empty();
      read_regst_.at(regst->regst_desc_id()).push(regst);
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return msg_handler() == nullptr;
}

int BoxingActor::HandlerWaitUntilNoReadableRegst(const ActorMsg& msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst()), 0);
  ActUntilFail();
  if (num_of_read_empty()) {
    AsyncSendEORDMsgForAllProducedRegstDesc();
    OF_SET_MSG_HANDLER(&BoxingActor::HandlerZombie);
  }
  return 0;
}

void BoxingActor::Act() {
  int64_t piece_id = expected_piece_id();
  for (const auto& pair : read_regst_) {
    CHECK_EQ(pair.second.front()->piece_id(), piece_id);
  }
  AsyncLaunchKernel(GenDefaultKernelCtx(),
                    [this](int64_t regst_desc_id) -> Regst* {
                      Regst* regst = GetCurWriteableRegst(regst_desc_id);
                      if (regst == nullptr) {
                        return read_regst_.at(regst_desc_id).front();
                      } else {
                        return regst;
                      }
                    });
  AsyncSendRegstMsgToConsumer(
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
