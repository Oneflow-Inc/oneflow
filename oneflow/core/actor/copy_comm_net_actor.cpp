#include "oneflow/core/actor/copy_comm_net_actor.h"
#include "oneflow/core/actor/actor_registry.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

void CopyCommNetActor::Init(const TaskProto& task_proto,
                            const ThreadCtx& thread_ctx) {
  Actor::Init(task_proto, thread_ctx);
  CHECK(thread_ctx.cpu_stream);
  set_num_of_remaining_eord(1);
  mut_num_of_read_empty() = 1;
  mut_device_ctx().reset(new CpuDeviceCtx(thread_ctx.cpu_stream));
  OF_SET_MSG_HANDLER(&CopyCommNetActor::HandlerNormal);
}

int CopyCommNetActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kEORD);
    ProcessEord();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(regst) != 0) {
      mut_num_of_read_empty() = 0;
      CHECK(
          piece_id2waiting_in_regst_.emplace(regst->piece_id(), regst).second);
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return msg_handler() == nullptr;
}

int CopyCommNetActor::HandlerWaitUntilNoReadableRegst(const ActorMsg& msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst()), 0);
  ActUntilFail();
  if (piece_id2waiting_in_regst_.empty()) {
    AsyncSendEORDMsgForAllProducedRegstDesc();
    OF_SET_MSG_HANDLER(&CopyCommNetActor::HandlerZombie);
  }
  return 0;
}

void CopyCommNetActor::Act() {
  auto next_regst_it = piece_id2waiting_in_regst_.find(expected_piece_id());
  Regst* next_regst = next_regst_it->second;
  AsyncLaunchKernel(GenDefaultKernelCtx(),
                    [&](uint64_t regst_desc_id) -> Regst* {
                      Regst* regst = GetCurWriteableRegst(regst_desc_id);
                      if (regst == nullptr) {
                        return next_regst;
                      } else {
                        return regst;
                      }
                    });
  AsyncSendRegstMsgToConsumer([&next_regst](Regst* regst) {
    regst->set_piece_id(next_regst->piece_id());
    regst->set_model_version_id(next_regst->model_version_id());
  });
  AsyncSendRegstMsgToProducer(next_regst);
  piece_id2waiting_in_regst_.erase(next_regst_it);
  mut_num_of_read_empty() = piece_id2waiting_in_regst_.empty();
}

REGISTER_ACTOR(kCopyCommNetTask, true, CopyCommNetActor);
REGISTER_ACTOR(kCopyCommNetTask, false, CopyCommNetActor);

}  // namespace oneflow
