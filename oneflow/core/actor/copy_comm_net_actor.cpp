#include "oneflow/core/actor/copy_comm_net_actor.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/job/machine_context.h"
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

  CommNetDeviceCtx(int64_t work_stream_id, void* actor_read_id)
      : DeviceCtx(), actor_read_id_(actor_read_id), read_id_(nullptr) {
    set_work_stream_id(work_stream_id);
  }

  void AddCallBack(std::function<void()> callback) const override {
    CommNet::Singleton()->AddReadCallBack(actor_read_id_, read_id_, callback);
  }

  void set_read_id(void* val) { read_id_ = val; }

 private:
  void* actor_read_id_;
  void* read_id_;
};

void CopyCommNetActor::VirtualActorInit(const TaskProto& task_proto) {
  is_in_eord_ = false;
  actor_read_id_ = CommNet::Singleton()->NewActorReadId();
  comm_net_device_ctx_ =
      new CommNetDeviceCtx(GetReservedWorkStreamId(0), actor_read_id_);
  next_piece_id_ = 0;
  OF_SET_MSG_HANDLER(&CopyCommNetActor::HandlerNormal);
}

void CopyCommNetActor::InitDeviceCtx(const ThreadCtx&) {
  mut_device_ctx().reset(comm_net_device_ctx_);
}

int CopyCommNetActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    DecreaseRemainingEordCnt();
    is_in_eord_ = true;
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (msg.SrcMachineId() == MachineCtx::Singleton()->this_machine_id()) {
      CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst()), 0);
    } else {
      RegstCtx regst_ctx;
      regst_ctx.comm_net_token = msg.comm_net_token();
      regst_ctx.regst_raw_ptr = msg.regst();
      regst_ctx.producer = msg.src_actor_id();
      regst_ctx.act_id = msg.act_id();
      CHECK(piece_id2regst_ctx.emplace(msg.piece_id(), regst_ctx).second);
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
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
  AsyncSendRegstMsgToConsumer([&](Regst* regst) {
    regst->set_piece_id(next_piece_id_);
    return true;
  });
  AsyncSendRegstMsgToProducer(readable_regst, src_actor_id);
  comm_net_device_ctx_->set_read_id(nullptr);
  CommNet::Singleton()->AddReadCallBackDone(read_id);
  piece_id2regst_ctx.erase(readable_it);
  next_piece_id_ += 1;
}

bool CopyCommNetActor::IsReadReady() {
  return piece_id2regst_ctx.find(next_piece_id_) != piece_id2regst_ctx.end();
}

bool CopyCommNetActor::IsReadAlwaysUnReadyFromNow() {
  return is_in_eord_ && piece_id2regst_ctx.empty();
}

void CopyCommNetActor::AsyncReturnAllReadableRegst() {
  CHECK(piece_id2regst_ctx.empty());
}

void CopyCommNetActor::ForEachCurReadableRegst(
    std::function<void(const Regst*)> SetRegInfo) {
  SetRegInfo(nullptr);
}

void CopyCommNetActor::SetReadableRegstInfo(const Regst* reg,
                                            ReadableRegstInfo* info) {
  CHECK(RegstDescId4Name("copy_in") != -1);
  info->set_regst_desc_id(RegstDescId4Name("copy_in"));
  info->set_act_id(piece_id2regst_ctx[next_piece_id_].act_id);
}

REGISTER_ACTOR(TaskType::kCopyCommNet, CopyCommNetActor);

}  // namespace oneflow
