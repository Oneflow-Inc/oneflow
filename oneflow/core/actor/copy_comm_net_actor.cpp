#include "oneflow/core/actor/copy_comm_net_actor.h"
#include "oneflow/core/actor/actor_registry.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

namespace {

class CommNetDeviceCtx final : public DeviceCtx {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(CommNetDeviceCtx);
  CommNetDeviceCtx() = delete;
  ~CommNetDeviceCtx() = default;

  CommNetDeviceCtx(void* stream_id) : DeviceCtx(), stream_id_(stream_id) {}

  void AddCallBack(std::function<void()> callback) const override {
    CommNet::Singleton()->AddCallBack(stream_id_, callback);
  }

 private:
  void* stream_id_;
};

}  // namespace

void CopyCommNetActor::Init(const TaskProto& task_proto,
                            const ThreadCtx& thread_ctx) {
  Actor::Init(task_proto, thread_ctx);
  set_num_of_remaining_eord(1);
  mut_num_of_read_empty() = 1;
  stream_id_ = CommNet::Singleton()->CreateStream();
  mut_device_ctx().reset(new CommNetDeviceCtx(stream_id_));
  OF_SET_MSG_HANDLER(&CopyCommNetActor::HandlerNormal);
}

int CopyCommNetActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kEORD);
    ProcessEord();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (msg.SrcMachineId() == RuntimeCtx::Singleton()->this_machine_id()) {
      CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst()), 0);
    } else {
      mut_num_of_read_empty() = 0;
      RegstCtx regst_ctx;
      regst_ctx.comm_net_token = msg.comm_net_token();
      regst_ctx.regst_raw_ptr = msg.regst();
      regst_ctx.producer = msg.src_actor_id();
      CHECK(piece_id2regst_ctx.emplace(msg.piece_id(), regst_ctx).second);
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
  if (piece_id2regst_ctx.empty()) {
    AsyncSendEORDMsgForAllProducedRegstDesc();
    OF_SET_MSG_HANDLER(&CopyCommNetActor::HandlerZombie);
  }
  return 0;
}

void CopyCommNetActor::Act() {
  int64_t cur_piece_id = expected_piece_id();
  // readable
  auto readable_it = piece_id2regst_ctx.find(cur_piece_id);
  const void* readable_token = readable_it->second.comm_net_token;
  Regst* readable_regst = readable_it->second.regst_raw_ptr;
  // writeable
  Regst* writeable_regst = GetCurSoleWriteableRegst();
  const void* writeable_token =
      writeable_regst->packed_blob()->comm_net_token();
  // Async
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  auto other_val = std::make_tuple(stream_id_, readable_token, writeable_token);
  kernel_ctx.other = &other_val;
  AsyncLaunchKernel(kernel_ctx);
  AsyncSendRegstMsgToConsumer(
      [&](Regst* regst) { regst->set_piece_id(cur_piece_id); });
  AsyncSendRegstMsgToProducer(readable_regst, readable_it->second.producer);
  // Finish
  piece_id2regst_ctx.erase(readable_it);
  mut_num_of_read_empty() = piece_id2regst_ctx.empty();
}

REGISTER_ACTOR(kCopyCommNetTask, true, CopyCommNetActor);
REGISTER_ACTOR(kCopyCommNetTask, false, CopyCommNetActor);

}  // namespace oneflow
