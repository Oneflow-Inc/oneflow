#ifndef ONEFLOW_CORE_ACTOR_CUDA_COPY_PEER_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_CUDA_COPY_PEER_ACTOR_H_

#include "oneflow/core/actor/actor.h"

namespace oneflow {

struct CudaCopyPeerCtx;

class CudaCopyPeerActor : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaCopyPeerActor);
  CudaCopyPeerActor() = default;
  ~CudaCopyPeerActor() override;

 protected:
  void VirtualActorInit(const TaskProto&) override;
  int HandlerCopy(const ActorMsg& msg);

  int64_t in_regst_desc_id_ = -1;
  std::deque<Regst*> in_regst_deque_;
  bool in_regst_eord_ = false;
  int64_t out_regst_desc_id_ = -1;
  Regst* out_regst_ = nullptr;
  int64_t out_regst_reading_cnt_ = -1;
  bool eord_sent_ = false;
  LogicalBlobId lbi_;
  int32_t dst_dev_id_ = -1;
  int32_t src_dev_id_ = -1;
  CudaCopyPeerCtx* cuda_copy_peer_ctx_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_CUDA_COPY_PEER_ACTOR_H_
