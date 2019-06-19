#include "oneflow/core/actor/local_ring_boxing_actor.h"

namespace oneflow {

void LocalRingBoxingActor::VirtualActorInit(const TaskProto& task_proto) {
  CHECK_EQ(1, exec_kernel_vec().size());

  out_regst_desc_id_ = task_proto.produced_regst_desc().at("out").regst_desc_id();
  send_regst_desc_id_ = task_proto.produced_regst_desc().at("send").regst_desc_id();
  in_regst_desc_id_ = task_proto.consumed_regst_desc_id().at("in").regst_desc_id(0);
  recv_regst_desc_id_ = task_proto.consumed_regst_desc_id().at("recv").regst_desc_id(0);

  current_step_ = 0;
  const OperatorConf& op_conf =
      task_proto.exec_sequence().exec_node(0).kernel_conf().op_attribute().op_conf();
  if (op_conf.has_local_ring_reduce_scatter_conf()) {
    total_num_steps_ =
        op_conf.local_ring_reduce_scatter_conf().local_ring_boxing_conf().ring_size();
  } else if (op_conf.has_local_ring_all_gather_conf()) {
    total_num_steps_ = op_conf.local_ring_all_gather_conf().local_ring_boxing_conf().ring_size();
  } else if (op_conf.has_local_ring_all_reduce_conf()) {
    total_num_steps_ =
        op_conf.local_ring_all_reduce_conf().local_ring_boxing_conf().ring_size() * 2 - 1;
  } else {
    UNIMPLEMENTED();
  }

  consumed_rs_.InsertRegstDescId(in_regst_desc_id_);
  consumed_rs_.InsertRegstDescId(recv_regst_desc_id_);
  consumed_rs_.InitedDone();

  OF_SET_MSG_HANDLER(&LocalRingBoxingActor::HandlerNormal);
}

int64_t LocalRingBoxingActor::ActNumForEachOutput(int64_t regst_desc_id) const {
  return total_num_steps_;
}

void LocalRingBoxingActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  CHECK_EQ(0, consumed_rs_.TryPushBackRegst(msg.regst()));
}

bool LocalRingBoxingActor::IsCustomizedReadReady() const {
  if (current_step_ == 0) {
    return consumed_rs_.Front(in_regst_desc_id_) != nullptr;
  } else {
    return consumed_rs_.Front(in_regst_desc_id_) != nullptr
           && consumed_rs_.Front(recv_regst_desc_id_) != nullptr;
  }
}

void LocalRingBoxingActor::ForEachCurCustomizedReadableRegst(
    std::function<void(const Regst*)> handler) const {
  if (current_step_ != 0) { handler(consumed_rs_.Front(recv_regst_desc_id_)); }
  handler(consumed_rs_.Front(in_regst_desc_id_));
}

void LocalRingBoxingActor::Act() {
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  SetKernelCtxOther(&(kernel_ctx.other));
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    return consumed_rs_.Front(regst_desc_id);
  });
}

void LocalRingBoxingActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  if (current_step_ == total_num_steps_ - 1) {
    HandleProducedNaiveDataRegstToConsumer([this](Regst* regst) {
      if (regst->regst_desc_id() == out_regst_desc_id_) {
        regst->set_piece_id(consumed_rs_.Front(in_regst_desc_id_)->piece_id());
        return true;
      } else {
        return false;
      }
    });
  } else {
    HandleProducedNaiveDataRegstToConsumer([this](Regst* regst) {
      if (regst->regst_desc_id() == send_regst_desc_id_) {
        regst->set_piece_id(consumed_rs_.Front(in_regst_desc_id_)->piece_id());
        return true;
      } else {
        return false;
      }
    });
  }
}

void LocalRingBoxingActor::AsyncSendCustomizedConsumedRegstMsgToProducer() {
  if (current_step_ == total_num_steps_ - 1) {
    Regst* cur_regst = consumed_rs_.Front(in_regst_desc_id_);
    CHECK(cur_regst);
    AsyncSendRegstMsgToProducer(cur_regst);
    CHECK_EQ(0, consumed_rs_.TryPopFrontRegst(in_regst_desc_id_));
  }
  if (current_step_ != 0) {
    Regst* cur_regst = consumed_rs_.Front(recv_regst_desc_id_);
    CHECK(cur_regst);
    AsyncSendRegstMsgToProducer(cur_regst);
    CHECK_EQ(0, consumed_rs_.TryPopFrontRegst(recv_regst_desc_id_));
  }
  current_step_ = (current_step_ + 1) % total_num_steps_;
}

void LocalRingBoxingActor::AsyncReturnAllCustomizedReadableRegst() {
  CHECK_EQ(0, consumed_rs_.available_regst_desc_cnt());
}

bool LocalRingBoxingActor::ProducedCtrlRegstValid(int64_t regst_desc_id) const { return true; }

REGISTER_ACTOR(kLocalRingBoxing, LocalRingBoxingActor);

}  // namespace oneflow
