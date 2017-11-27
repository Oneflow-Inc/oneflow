#include "oneflow/core/actor/copy_comm_net_actor.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

CopyCommNetActor::~CopyCommNetActor() {
  CommNet::Singleton()->DeleteActorReadId(actor_read_id_);
}

class CopyCommNetActor::CommNetDeviceCtx final : public DeviceCtx {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(CommNetDeviceCtx);
  CommNetDeviceCtx() = delete;
  ~CommNetDeviceCtx() = default;

  CommNetDeviceCtx(void* actor_read_id)
      : DeviceCtx(), actor_read_id_(actor_read_id), read_id_(nullptr) {}

  void AddCallBack(std::function<void()> callback) const override {
    CommNet::Singleton()->AddReadCallBack(actor_read_id_, read_id_, callback);
  }

  void set_read_id(void* val) { read_id_ = val; }

 private:
  void* actor_read_id_;
  void* read_id_;
};

void CopyCommNetActor::VirtualActorInit(const TaskProto& task_proto) {
  actor_read_id_ = CommNet::Singleton()->NewActorReadId();
  comm_net_device_ctx_ = new CommNetDeviceCtx(actor_read_id_);
  next_piece_id_ = 0;
  OF_SET_MSG_HANDLER(&CopyCommNetActor::HandlerNormal);
}

void CopyCommNetActor::InitDeviceCtx(const ThreadCtx&) {
  mut_device_ctx().reset(comm_net_device_ctx_);
}

int CopyCommNetActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    return 1;
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (msg.SrcMachineId() == RuntimeCtx::Singleton()->this_machine_id()) {
      CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst()), 0);
    } else {
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
  return 0;
}

void CopyCommNetActor::Act() {
  // readable
  auto readable_it = piece_id2regst_ctx.find(next_piece_id_);
  const void* readable_token = readable_it->second.comm_net_token;
  Regst* readable_regst = readable_it->second.regst_raw_ptr;
  int64_t src_actor_id = readable_it->second.producer;
  int64_t src_machine_id = IDMgr::Singleton()->MachineId4ActorId(src_actor_id);
  // writeable
  Blob* writeable_blob = GetCurSoleWriteableRegst()->packed_blob();
  const void* writeable_token = writeable_blob->comm_net_token();
  // Async
  void* read_id = CommNet::Singleton()->Read(actor_read_id_, src_machine_id,
                                             readable_token, writeable_token);
  comm_net_device_ctx_->set_read_id(read_id);
  AsyncSendRegstMsgToConsumer(
      [&](Regst* regst) { regst->set_piece_id(next_piece_id_); });
  AsyncSendRegstMsgToProducer(readable_regst, src_actor_id);
  comm_net_device_ctx_->set_read_id(nullptr);
  CommNet::Singleton()->AddReadCallBackDone(actor_read_id_, read_id);
  piece_id2regst_ctx.erase(readable_it);
  next_piece_id_ += 1;
}

bool CopyCommNetActor::IsReadReady() {
  return piece_id2regst_ctx.find(next_piece_id_) != piece_id2regst_ctx.end();
}

bool CopyCommNetActor::IsReadAlwaysUnReadyFromNow() {
  UNEXPECTED_RUN();
  return false;
}

void CopyCommNetActor::AsyncReturnAllReadableRegst() { UNEXPECTED_RUN(); }

REGISTER_ACTOR(TaskType::kCopyCommNet, CopyCommNetActor);

}  // namespace oneflow
