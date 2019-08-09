#include "oneflow/core/actor/cuda_ring_all_reduce_compute_actor.h"

namespace oneflow {

void CudaRingAllReduceCompActor::VirtualActorInit(const TaskProto& task_proto) {
  CHECK_EQ(1, exec_kernel_vec().size());
  const CudaRingAllReduceOpConf& conf =
      exec_kernel_vec().front().kernel->op_attribute().op_conf().cuda_ring_all_reduce_conf();
  num_link_ = conf.link_size();
  CHECK_GT(num_link_, 0);
  out_regst_desc_id_ = task_proto.produced_regst_desc().at("out").regst_desc_id();
  const RegstDescIdSet& consumed_in_regst_desc_ids = task_proto.consumed_regst_desc_id().at("in");
  CHECK_EQ(consumed_in_regst_desc_ids.regst_desc_id_size(), 1);
  in_regst_desc_id_ = consumed_in_regst_desc_ids.regst_desc_id(0);
  consumed_rs_.InsertRegstDescId(in_regst_desc_id_);
  FOR_RANGE(int64_t, i, 0, num_link_) {
    const int64_t send_regst_desc_id =
        task_proto.produced_regst_desc().at("send_" + std::to_string(i)).regst_desc_id();
    const RegstDescIdSet& consumed_recv_regst_desc_ids =
        task_proto.consumed_regst_desc_id().at("recv_" + std::to_string(i));
    CHECK_EQ(consumed_recv_regst_desc_ids.regst_desc_id_size(), 1);
    const int64_t recv_regst_desc_id = consumed_recv_regst_desc_ids.regst_desc_id(0);
    send_regst_desc_ids_.emplace(send_regst_desc_id);
    recv_regst_desc_ids_.emplace(recv_regst_desc_id);
    consumed_rs_.InsertRegstDescId(recv_regst_desc_id);
  }
  num_step_ = parallel_ctx()->parallel_num() * 2 - 1;
  slice_factor_ = conf.slice_factor();
  current_step_id_ = 0;
  current_slice_id_ = 0;
  send_regst_piece_id_ = 0;
  consumed_rs_.InitedDone();
  in_regst_eord_ = false;
  OF_SET_MSG_HANDLER(&CudaRingAllReduceCompActor::HandlerNormal);
}

int64_t CudaRingAllReduceCompActor::ActNumForEachOutput(int64_t regst_desc_id) const {
  if (regst_desc_id == out_regst_desc_id_) {
    return num_step_ * slice_factor_;
  } else if (send_regst_desc_ids_.find(regst_desc_id) != send_regst_desc_ids_.cend()) {
    return 1;
  } else {
    UNIMPLEMENTED();
  }
}

void CudaRingAllReduceCompActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  CHECK_EQ(0, consumed_rs_.TryPushBackRegst(msg.regst()));
}

void CudaRingAllReduceCompActor::NormalProcessCustomizedEordMsg(const ActorMsg& msg) {
  if (msg.eord_regst_desc_id() == in_regst_desc_id_) { in_regst_eord_ = true; }
}

bool CudaRingAllReduceCompActor::IsCustomizedReadAlwaysUnReadyFromNow() const {
  return in_regst_eord_ && consumed_rs_.available_regst_desc_cnt() == 0 && current_step_id_ == 0
         && current_slice_id_ == 0;
}

bool CudaRingAllReduceCompActor::IsCustomizedReadReady() const {
  if (current_step_id_ == 0) {
    return consumed_rs_.Front(in_regst_desc_id_) != nullptr;
  } else {
    return consumed_rs_.available_regst_desc_cnt() == consumed_rs_.total_regst_desc_cnt();
  }
}

void CudaRingAllReduceCompActor::ForEachCurCustomizedReadableRegst(
    std::function<void(const Regst*)> handler) const {
  if (current_step_id_ != 0) {
    for (const int64_t recv_regst_desc_id : recv_regst_desc_ids_) {
      handler(consumed_rs_.Front(recv_regst_desc_id));
    }
  }
  handler(consumed_rs_.Front(in_regst_desc_id_));
}

void CudaRingAllReduceCompActor::Act() {
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  SetKernelCtxOther(&(kernel_ctx.other));
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    return consumed_rs_.Front(regst_desc_id);
  });
}

void CudaRingAllReduceCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  if (current_step_id_ == num_step_ - 1 && current_slice_id_ == slice_factor_ - 1) {
    HandleProducedNaiveDataRegstToConsumer([this](Regst* regst) {
      if (regst->regst_desc_id() == out_regst_desc_id_) {
        regst->set_piece_id(consumed_rs_.Front(in_regst_desc_id_)->piece_id());
        return true;
      } else {
        return false;
      }
    });
  } else if (current_step_id_ < num_step_ - 1) {
    HandleProducedNaiveDataRegstToConsumer([this](Regst* regst) {
      if (send_regst_desc_ids_.find(regst->regst_desc_id()) != send_regst_desc_ids_.cend()) {
        regst->set_piece_id(send_regst_piece_id_);
        return true;
      } else {
        return false;
      }
    });
    send_regst_piece_id_ += 1;
  }
}

void CudaRingAllReduceCompActor::AsyncSendCustomizedConsumedRegstMsgToProducer() {
  if (current_step_id_ == num_step_ - 1 && current_slice_id_ == slice_factor_ - 1) {
    Regst* cur_regst = consumed_rs_.Front(in_regst_desc_id_);
    CHECK(cur_regst);
    AsyncSendRegstMsgToProducer(cur_regst);
    CHECK_EQ(0, consumed_rs_.TryPopFrontRegst(in_regst_desc_id_));
  }
  if (current_step_id_ != 0) {
    for (const int64_t recv_regst_desc_id : recv_regst_desc_ids_) {
      Regst* cur_regst = consumed_rs_.Front(recv_regst_desc_id);
      CHECK(cur_regst);
      AsyncSendRegstMsgToProducer(cur_regst);
      CHECK_EQ(0, consumed_rs_.TryPopFrontRegst(recv_regst_desc_id));
    }
  }
  current_slice_id_ = (current_slice_id_ + 1) % slice_factor_;
  if (current_slice_id_ == 0) { current_step_id_ = (current_step_id_ + 1) % num_step_; }
}

bool CudaRingAllReduceCompActor::CheckOutputActId(int64_t regst_desc_id) const {
  if (regst_desc_id == out_regst_desc_id_) {
    return true;
  } else if (send_regst_desc_ids_.find(regst_desc_id) != send_regst_desc_ids_.cend()) {
    return false;
  } else {
    UNIMPLEMENTED();
  }
}

void CudaRingAllReduceCompActor::SetKernelCtxOther(void** other) {
  other_ctx_.first = current_step_id_;
  other_ctx_.second = current_slice_id_;
  *other = &other_ctx_;
}

void CudaRingAllReduceCompActor::AsyncReturnAllCustomizedReadableRegst() {
  CHECK_EQ(0, consumed_rs_.available_regst_desc_cnt());
}

bool CudaRingAllReduceCompActor::ProducedCtrlRegstValid(int64_t regst_desc_id) const {
  return true;
}

REGISTER_ACTOR(kCudaRingAllReduce, CudaRingAllReduceCompActor);

}  // namespace oneflow
