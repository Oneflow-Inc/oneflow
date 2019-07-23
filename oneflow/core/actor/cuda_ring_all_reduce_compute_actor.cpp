#include "oneflow/core/actor/cuda_ring_all_reduce_compute_actor.h"

namespace oneflow {

void CudaRingAllReduceCompActor::VirtualActorInit(const TaskProto& task_proto) {
  CHECK_EQ(1, exec_kernel_vec().size());
  const OperatorConf& op_conf =
      exec_kernel_vec().front().kernel->kernel_conf().op_attribute().op_conf();
  CHECK(op_conf.has_cuda_ring_all_reduce_conf());
  const CudaRingAllReduceOpConf& conf = op_conf.cuda_ring_all_reduce_conf();
  num_rings_ = conf.rings_size();
  CHECK_GT(num_rings_, 0);
  out_regst_desc_id_ = task_proto.produced_regst_desc().at("out").regst_desc_id();
  in_regst_desc_id_ = task_proto.consumed_regst_desc_id().at("in").regst_desc_id(0);
  FOR_RANGE(int64_t, i, 0, num_rings_) {
    const int64_t send_regst_desc_id = task_proto.produced_regst_desc().at("send").regst_desc_id();
    const int64_t recv_regst_desc_id =
        task_proto.consumed_regst_desc_id().at("recv").regst_desc_id(0);
    send_regst_desc_ids_.emplace(send_regst_desc_id);
    recv_regst_desc_ids_.emplace(recv_regst_desc_id);
    consumed_rs_.InsertRegstDescId(recv_regst_desc_id);
  }
  total_num_steps_ = conf.rings(0).next_size();
  FOR_RANGE(int64_t, i, 1, num_rings_) { CHECK_EQ(conf.rings(i).next_size(), total_num_steps_); }
  current_step_ = 0;
  send_regst_piece_id_ = 0;
  consumed_rs_.InsertRegstDescId(in_regst_desc_id_);
  consumed_rs_.InitedDone();
  in_regst_eord_ = false;
  OF_SET_MSG_HANDLER(&CudaRingAllReduceCompActor::HandlerNormal);
}

int64_t CudaRingAllReduceCompActor::ActNumForEachOutput(int64_t regst_desc_id) const {
  if (regst_desc_id == out_regst_desc_id_) {
    return total_num_steps_;
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
  return in_regst_eord_ && consumed_rs_.available_regst_desc_cnt() == 0 && current_step_ == 0;
}

bool CudaRingAllReduceCompActor::IsCustomizedReadReady() const {
  if (current_step_ == 0) {
    return consumed_rs_.Front(in_regst_desc_id_) != nullptr;
  } else {
    return consumed_rs_.available_regst_desc_cnt() == consumed_rs_.total_regst_desc_cnt();
  }
}

void CudaRingAllReduceCompActor::ForEachCurCustomizedReadableRegst(
    std::function<void(const Regst*)> handler) const {
  if (current_step_ != 0) {
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
  if (current_step_ == total_num_steps_ - 1) {
    Regst* cur_regst = consumed_rs_.Front(in_regst_desc_id_);
    CHECK(cur_regst);
    AsyncSendRegstMsgToProducer(cur_regst);
    CHECK_EQ(0, consumed_rs_.TryPopFrontRegst(in_regst_desc_id_));
  }
  if (current_step_ != 0) {
    for (const int64_t recv_regst_desc_id : recv_regst_desc_ids_) {
      Regst* cur_regst = consumed_rs_.Front(recv_regst_desc_id);
      CHECK(cur_regst);
      AsyncSendRegstMsgToProducer(cur_regst);
      CHECK_EQ(0, consumed_rs_.TryPopFrontRegst(recv_regst_desc_id));
    }
  }
  current_step_ = (current_step_ + 1) % total_num_steps_;
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

void CudaRingAllReduceCompActor::SetKernelCtxOther(void** other) { *other = &current_step_; }

void CudaRingAllReduceCompActor::AsyncReturnAllCustomizedReadableRegst() {
  CHECK_EQ(0, consumed_rs_.available_regst_desc_cnt());
}

bool CudaRingAllReduceCompActor::ProducedCtrlRegstValid(int64_t regst_desc_id) const {
  return true;
}

REGISTER_ACTOR(kCudaRingAllReduce, CudaRingAllReduceCompActor);

}  // namespace oneflow
