#include "oneflow/core/actor/multi_ring_all_reduce_actor.h"

namespace oneflow {

void MultiRingAllReduceActor::VirtualActorInit(const TaskProto& task_proto) {
  CHECK_EQ(1, exec_kernel_vec().size());
  out_regst_desc_id_ = task_proto.produced_regst_desc().at("out").regst_desc_id();
  produced_rs_.InsertRegstDescId(out_regst_desc_id_);
  CHECK_EQ(task_proto.consumed_regst_desc_id().at("in").regst_desc_id_size(), 1);
  in_regst_desc_id_ = task_proto.consumed_regst_desc_id().at("in").regst_desc_id(0);
  consumed_rs_.InsertRegstDescId(in_regst_desc_id_);
  const MultiRingAllReduceOpConf& conf = task_proto.exec_sequence()
                                             .exec_node(0)
                                             .kernel_conf()
                                             .op_attribute()
                                             .op_conf()
                                             .multi_ring_all_reduce_conf();
  num_rings_ = conf.rings_size();
  CHECK_GE(num_rings_, 1);
  num_steps_ = conf.rings(0).next_size() * 2 - 1;
  FOR_RANGE(int64_t, ring_id, 0, num_rings_) {
    current_step_.push_back(0);
    const std::string send_name = "send_" + std::to_string(ring_id);
    const std::string recv_name = "recv_" + std::to_string(ring_id);
    const int64_t send_regst_desc_id =
        task_proto.produced_regst_desc().at(send_name).regst_desc_id();
    send_regst_desc_id_.push_back(send_regst_desc_id);
    send_regst_piece_id_.push_back(0);
    produced_rs_.InsertRegstDescId(send_regst_desc_id);
    CHECK_EQ(task_proto.consumed_regst_desc_id().at(recv_name).regst_desc_id_size(), 0);
    const int64_t recv_regst_desc_id =
        task_proto.consumed_regst_desc_id().at(recv_name).regst_desc_id(0);
    recv_regst_desc_id_.push_back(recv_regst_desc_id);
    consumed_rs_.InsertRegstDescId(recv_regst_desc_id);
  }
  produced_rs_.InitedDone();
  consumed_rs_.InitedDone();
  in_regst_eord_ = false;
  OF_SET_MSG_HANDLER(&MultiRingAllReduceActor::HandlerNormal);
}

int64_t MultiRingAllReduceActor::ActNumForEachOutput(int64_t regst_desc_id) const {
  if (regst_desc_id == out_regst_desc_id_) {
    return num_steps_;
  } else {
    return 1;
  }
}

void MultiRingAllReduceActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  CHECK_EQ(0, consumed_rs_.TryPushBackRegst(msg.regst()));
}

void MultiRingAllReduceActor::NormalProcessCustomizedEordMsg(const ActorMsg& msg) {
  if (msg.eord_regst_desc_id() == in_regst_desc_id_) { in_regst_eord_ = true; }
}

bool MultiRingAllReduceActor::IsCustomizedReadAlwaysUnReadyFromNow() const {
  return in_regst_eord_ && consumed_rs_.available_regst_desc_cnt() == 0
         && std::all_of(current_step_.cbegin(), current_step_.cend(),
                        [](const int64_t current_step) { return current_step == 0; });
}

bool MultiRingAllReduceActor::IsCustomizedReadReady() const { return GetCurReadyRing() != -1; }

bool MultiRingAllReduceActor::IsCustomizedWriteReady() const { return GetCurReadyRing() != -1; }

void MultiRingAllReduceActor::ForEachCurCustomizedReadableRegst(
    std::function<void(const Regst*)> handler) const {
  const int64_t step = current_step_.at(current_ring_id_);
  if (step != 0) { handler(consumed_rs_.Front(recv_regst_desc_id_.at(current_ring_id_))); }
  handler(consumed_rs_.Front(in_regst_desc_id_));
}

void MultiRingAllReduceActor::Act() {
  current_ring_id_ = GetCurReadyRing();
  CHECK_NE(current_ring_id_, -1);
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  SetKernelCtxOther(&(kernel_ctx.other));
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    return consumed_rs_.Front(regst_desc_id);
  });
}

void MultiRingAllReduceActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  const int64_t step = current_step_.at(current_ring_id_);
  if (step == num_steps_ - 1) {
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
      if (regst->regst_desc_id() == send_regst_desc_id_.at(current_ring_id_)) {
        regst->set_piece_id(send_regst_piece_id_.at(current_ring_id_));
        send_regst_piece_id_[current_ring_id_] += 1;
        return true;
      } else {
        return false;
      }
    });
  }
}

void MultiRingAllReduceActor::AsyncSendCustomizedConsumedRegstMsgToProducer() {
  const int64_t step = current_step_.at(current_ring_id_);
  if (step == num_steps_ - 1) {
    Regst* cur_regst = consumed_rs_.Front(in_regst_desc_id_);
    CHECK(cur_regst);
    AsyncSendRegstMsgToProducer(cur_regst);
    CHECK_EQ(0, consumed_rs_.TryPopFrontRegst(in_regst_desc_id_));
  }
  if (step != 0) {
    Regst* cur_regst = consumed_rs_.Front(recv_regst_desc_id_.at(current_ring_id_));
    CHECK(cur_regst);
    AsyncSendRegstMsgToProducer(cur_regst);
    CHECK_EQ(0, consumed_rs_.TryPopFrontRegst(recv_regst_desc_id_.at(current_ring_id_)));
  }
  current_step_[current_ring_id_] = (current_step_.at(current_ring_id_) + 1) % num_steps_;
}

bool MultiRingAllReduceActor::CheckOutputActId(int64_t regst_desc_id) const {
  if (regst_desc_id == out_regst_desc_id_) {
    return true;
  } else {
    return false;
  }
}

void MultiRingAllReduceActor::SetKernelCtxOther(void** other) {
  other_ctx_.first = current_ring_id_;
  other_ctx_.second = current_step_.at(current_ring_id_);
  *other = &other_ctx_;
}

int64_t MultiRingAllReduceActor::GetCurReadyRing() const {
  FOR_RANGE(int64_t, ring_id, 0, num_rings_) {
    if (IsRingReadReady(ring_id) && IsRingWriteReady(ring_id)) { return ring_id; }
  }
  return -1;
}

bool MultiRingAllReduceActor::IsRingReadReady(int64_t ring_id) const {
  const int64_t step = current_step_.at(ring_id);
  if (step == 0) {
    return consumed_rs_.Front(in_regst_desc_id_) != nullptr;
  } else {
    return consumed_rs_.Front(in_regst_desc_id_) != nullptr
           && consumed_rs_.Front(recv_regst_desc_id_.at(ring_id)) != nullptr;
  }
}

bool MultiRingAllReduceActor::IsRingWriteReady(int64_t ring_id) const {
  return produced_rs_.Front(out_regst_desc_id_) != nullptr
         && produced_rs_.Front(send_regst_desc_id_.at(ring_id)) != nullptr;
}

void MultiRingAllReduceActor::AsyncReturnAllCustomizedReadableRegst() {
  CHECK_EQ(0, consumed_rs_.available_regst_desc_cnt());
}

bool MultiRingAllReduceActor::ProducedCtrlRegstValid(int64_t regst_desc_id) const { return true; }

REGISTER_ACTOR(kMultiRingAllReduce, MultiRingAllReduceActor);

}  // namespace oneflow
