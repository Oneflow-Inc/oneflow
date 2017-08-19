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
      CHECK(piece_id2waiting_in_regst_.emplace(msg.piece_id(), regst).second);
      if (producer_actor_id_ == -1) {
        producer_actor_id_ = msg.src_actor_id();
      } else {
        CHECK(producer_actor_id_ == msg.src_actor_id());
      }
    }
    ActUntilFail();
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
  // TODO(jiyuan)
  auto next_regst_it = piece_id2waiting_in_regst_.find(expected_piece_id());
  Regst* in_regst = next_regst_it->second;
  LOG(INFO) << "CopyCommNetActor: " << actor_id();
  LOG(INFO) << "Read from: " << producer_actor_id_;

  // AsyncLaunchKernel(GenDefaultKernelCtx(),
  //                  [&](uint64_t regst_desc_id) -> Regst* {
  //                    Regst* regst = GetCurWriteableRegst(regst_desc_id);
  //                    if (regst == nullptr) {
  //                      return in_regst;
  //                    } else {
  //                      return regst;
  //                    }
  //                  });
  // AsyncSendReadableRegstMsg([&in_regst](Regst* regst) {
  //  regst->set_piece_id(in_regst->piece_id());
  //  regst->set_model_version_id(in_regst->model_version_id());
  //});
  // AsyncSendRegstMsgToProducer(in_regst);

  int64_t piece_id = expected_piece_id();
  int64_t model_version_id = 0;
  AsyncSendReadableRegstMsg([piece_id, model_version_id](Regst* regst) {
    regst->set_piece_id(piece_id);
    regst->set_model_version_id(model_version_id);
  });

  // AsyncSendRegstMsgToProducer(in_regst);
  ActorMsg msg = ActorMsg::BuildRegstMsgToProducer(producer_actor_id_,
                                                   actor_id(), in_regst);
  AsyncDo([msg]() { ActorMsgBus::Singleton()->SendMsg(msg); });
  // ActorMsgBus::Singleton()->SendMsg(msg);

  piece_id2waiting_in_regst_.erase(next_regst_it);
  mut_num_of_read_empty() = piece_id2waiting_in_regst_.empty();
}

REGISTER_ACTOR(kCopyCommNetTask, true, CopyCommNetActor);
REGISTER_ACTOR(kCopyCommNetTask, false, CopyCommNetActor);

}  // namespace oneflow
