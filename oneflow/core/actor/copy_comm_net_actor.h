#ifndef ONEFLOW_CORE_ACTOR_COPY_COMM_NET_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_COPY_COMM_NET_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

class CopyCommNetActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CopyCommNetActor);
  CopyCommNetActor() = default;
  ~CopyCommNetActor();

 private:
  class CommNetDeviceCtx;
  struct RegstCtx {
    const void* comm_net_token;
    Regst* regst_raw_ptr;
    int64_t producer;
    int64_t act_id;
  };

  void InitDeviceCtx(const ThreadCtx&) override;
  void VirtualActorInit(const TaskProto&) override;

  int HandlerNormal(const ActorMsg&) override;

  void Act() override;
  bool IsReadReady() override;
  bool IsReadAlwaysUnReadyFromNow() override;
  void AsyncReturnAllReadableRegst() override;

  void ForEachCurReadableRegst(std::function<void(const Regst*)>) override;
  void SetReadableRegstInfo(const Regst*, ReadableRegstInfo*) override;

  bool is_in_eord_;
  HashMap<int64_t, RegstCtx> piece_id2regst_ctx;
  void* actor_read_id_;
  CommNetDeviceCtx* comm_net_device_ctx_;
  int64_t next_piece_id_;
  int64_t in_regst_desc_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_COPY_COMM_NET_ACTOR_H_
