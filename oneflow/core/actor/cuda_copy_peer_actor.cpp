#include "oneflow/core/actor/cuda_copy_peer_actor.h"
#include "oneflow/core/kernel/multi_ring_all_reduce_kernel_util.h"

namespace oneflow {

int CudaCopyPeerActor::HandlerCopy(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    if (msg.eord_regst_desc_id() == in_regst_desc_id_) {
      in_regst_eord_ = true;
    } else {
      UNIMPLEMENTED();
    }
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    const int64_t regst_desc_id = msg.regst_desc_id();
    if (regst_desc_id == in_regst_desc_id_) {
      in_regst_deque_.push_back(msg.regst());
    } else if (regst_desc_id == out_regst_desc_id_) {
      CHECK_GT(out_regst_reading_cnt_, 0);
      out_regst_reading_cnt_ -= 1;
    } else {
      UNIMPLEMENTED();
    }
  } else {
    UNIMPLEMENTED();
  }
  while (!in_regst_deque_.empty() && out_regst_reading_cnt_ == 0) {
    Regst* current_in_regst = in_regst_deque_.front();
    const Blob* in_blob = current_in_regst->GetBlobByLbi(lbi_);
    Blob* out_blob = out_regst_->GetBlobByLbi(lbi_);
    current_in_regst->regst_desc()->mem_case().device_cuda_mem().device_id();
    DeviceCtx* device_ctx = mut_device_ctx().get();
    CudaCheck(cudaMemcpyAsync(out_blob->mut_dptr(), in_blob->dptr(),
                              in_blob->ByteSizeOfDataContentField(), cudaMemcpyDeviceToDevice,
                              device_ctx->cuda_stream()));
    std::vector<ActorMsg> actor_msgs_;
    out_regst_->set_piece_id(current_in_regst->piece_id());
    for (const int64_t consumer : out_regst_->consumers_actor_id()) {
      actor_msgs_.push_back(ActorMsg::BuildRegstMsgToConsumer(actor_id(), consumer, out_regst_));
    }
    out_regst_reading_cnt_ = out_regst_->consumers_actor_id().size();
    actor_msgs_.push_back(ActorMsg::BuildRegstMsgToProducer(
        actor_id(), current_in_regst->producer_actor_id(), current_in_regst));
    in_regst_deque_.pop_front();
    AsyncDo([actor_msgs_]() {
      for (const auto& msg : actor_msgs_) { Global<ActorMsgBus>::Get()->SendMsg(msg); }
    });
  }
  if (in_regst_eord_ && in_regst_deque_.empty()) {
    if (!eord_sent_) {
      std::vector<ActorMsg> actor_msgs_;
      for (const int64_t consumer : out_regst_->consumers_actor_id()) {
        actor_msgs_.push_back(ActorMsg::BuildEordMsg(consumer, out_regst_->regst_desc_id()));
      }
      AsyncDo([actor_msgs_]() {
        for (const auto& msg : actor_msgs_) { Global<ActorMsgBus>::Get()->SendMsg(msg); }
      });
      eord_sent_ = true;
    }
    if (eord_sent_ && out_regst_reading_cnt_ == 0) {
      OF_SET_MSG_HANDLER(nullptr);
      return 1;
    }
  }
  return 0;
}

void CudaCopyPeerActor::VirtualActorInit(const TaskProto& task_proto) {
  out_regst_desc_id_ = task_proto.produced_regst_desc().at("copy_out").regst_desc_id();
  out_regst_ = GetSoleProducedRegst4RegstDescId(out_regst_desc_id_);
  out_regst_reading_cnt_ = 0;
  CHECK_EQ(task_proto.consumed_regst_desc_id().at("copy_in").regst_desc_id_size(), 1);
  in_regst_desc_id_ = task_proto.consumed_regst_desc_id().at("copy_in").regst_desc_id(0);
  in_regst_eord_ = false;
  eord_sent_ = false;
  lbi_ = GenPackedLbi();
  OF_SET_MSG_HANDLER(&CudaCopyPeerActor::HandlerCopy);
}

REGISTER_ACTOR(kCudaCopyPeer, CudaCopyPeerActor);

}  // namespace oneflow
