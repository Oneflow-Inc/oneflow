#include "oneflow/core/actor/multi_ring_all_reduce_actor.h"
#include "oneflow/core/kernel/multi_ring_all_reduce_kernel_util.h"

namespace oneflow {

int MultiRingAllReduceActor::HandlerAllReduce(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    CHECK_EQ(msg.regst_desc_id(), in_regst_desc_id_);
    in_regst_eord_ = true;
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    const int64_t regst_desc_id = msg.regst_desc_id();
    if (regst_desc_id == in_regst_desc_id_) {
      in_regst_deque_.push_back(msg.regst());
    } else if (regst_desc_id == out_regst_desc_id_) {
      CHECK_GT(out_regst_desc_reading_cnt_, 0);
      out_regst_desc_reading_cnt_ -= 1;
    } else {
      const auto it = regst_desc_id2send_or_recv7ring_id_.find(regst_desc_id);
      CHECK(it != regst_desc_id2send_or_recv7ring_id_.cend());
      const bool is_send = it->second.first;
      const bool ring_id = it->second.second;
      if (is_send) {
        CHECK(!send_regst_ready_.at(ring_id));
        send_regst_ready_[ring_id] = true;
      } else {
        CHECK(!recv_regst_ready_.at(ring_id));
        recv_regst_ready_[ring_id] = true;
      }
    }
  } else {
    UNIMPLEMENTED();
  }
  const MultiRingAllReduceKernelStepConf& step_conf =
      multi_ring_all_reduce_kernel_conf_.ring_conf(current_ring_id_).step_conf(current_step_id_);
  if (!in_regst_deque_.empty() && out_regst_desc_reading_cnt_ == 0
      && (!step_conf.send() || send_regst_ready_.at(current_ring_id_))
      && (!step_conf.recv() || recv_regst_ready_.at(current_ring_id_))) {
  }
}

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
    const std::string send_name = "send_" + std::to_string(ring_id);
    const std::string recv_name = "recv_" + std::to_string(ring_id);
    const int64_t send_regst_desc_id =
        task_proto.produced_regst_desc().at(send_name).regst_desc_id();
    send_regst_desc_id_.push_back(send_regst_desc_id);
    send_regst_piece_id_.push_back(0);
    produced_rs_.InsertRegstDescId(send_regst_desc_id);
    CHECK_EQ(task_proto.consumed_regst_desc_id().at(recv_name).regst_desc_id_size(), 1);
    const int64_t recv_regst_desc_id =
        task_proto.consumed_regst_desc_id().at(recv_name).regst_desc_id(0);
    recv_regst_desc_id_.push_back(recv_regst_desc_id);
    consumed_rs_.InsertRegstDescId(recv_regst_desc_id);
  }
  produced_rs_.InitedDone();
  produced_rs_.TryPushBackRegst(GetSoleProducedRegst4RegstDescId(out_regst_desc_id_));
  for (const int64_t send_regst_desc_id : send_regst_desc_id_) {
    produced_rs_.TryPushBackRegst(GetSoleProducedRegst4RegstDescId(send_regst_desc_id));
  }
  consumed_rs_.InitedDone();
  in_regst_eord_ = false;
  current_ring_id_ = 0;
  current_step_id_ = 0;
  OF_SET_MSG_HANDLER(&MultiRingAllReduceActor::HandlerNormal);
}

int64_t MultiRingAllReduceActor::ActNumForEachOutput(int64_t regst_desc_id) const {
  if (regst_desc_id == out_regst_desc_id_) {
    return num_steps_ * num_rings_;
  } else {
    return 1;
  }
}

void MultiRingAllReduceActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  CHECK_EQ(0, consumed_rs_.TryPushBackRegst(msg.regst()));
}

void MultiRingAllReduceActor::UpdtStateAsCustomizedProducedRegst(Regst* regst) {
  CHECK_EQ(0, produced_rs_.TryPushBackRegst(regst));
}

void MultiRingAllReduceActor::NormalProcessCustomizedEordMsg(const ActorMsg& msg) {
  if (msg.eord_regst_desc_id() == in_regst_desc_id_) { in_regst_eord_ = true; }
}

bool MultiRingAllReduceActor::IsCustomizedReadAlwaysUnReadyFromNow() const {
  return in_regst_eord_ && consumed_rs_.available_regst_desc_cnt() == 0 && current_step_id_ == 0
         && current_ring_id_ == 0;
}

bool MultiRingAllReduceActor::IsCustomizedReadReady() const {
  if (current_step_id_ == 0) {
    return consumed_rs_.Front(in_regst_desc_id_) != nullptr;
  } else {
    return consumed_rs_.Front(in_regst_desc_id_) != nullptr
           && consumed_rs_.Front(recv_regst_desc_id_.at(current_ring_id_)) != nullptr;
  }
}

bool MultiRingAllReduceActor::IsCustomizedWriteReady() const {
  return produced_rs_.Front(out_regst_desc_id_) != nullptr
         && produced_rs_.Front(send_regst_desc_id_.at(current_ring_id_)) != nullptr;
}

void MultiRingAllReduceActor::ForEachCurCustomizedReadableRegst(
    std::function<void(const Regst*)> handler) const {
  if (current_step_id_ != 0) {
    handler(consumed_rs_.Front(recv_regst_desc_id_.at(current_ring_id_)));
  }
  handler(consumed_rs_.Front(in_regst_desc_id_));
}

void MultiRingAllReduceActor::Act() {
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  SetKernelCtxOther(&(kernel_ctx.other));
  AsyncLaunchKernel(kernel_ctx, [&](int64_t regst_desc_id) -> Regst* {
    if (produced_rs_.HasRegstDescId(regst_desc_id)) {
      return produced_rs_.Front(regst_desc_id);
    } else if (consumed_rs_.HasRegstDescId(regst_desc_id)) {
      return consumed_rs_.Front(regst_desc_id);
    } else {
      return nullptr;
    }
  });
}

void MultiRingAllReduceActor::AsyncSendCustomizedProducedRegstMsgToConsumer() {
  if (current_step_id_ == num_steps_ - 1 && current_ring_id_ == num_rings_ - 1) {
    Regst* out_regst = produced_rs_.Front(out_regst_desc_id_);
    out_regst->set_piece_id(consumed_rs_.Front(in_regst_desc_id_)->piece_id());
    HandleRegstToConsumer(out_regst, [](int64_t) { return true; });
    produced_rs_.PopFrontRegsts({out_regst_desc_id_});
  } else if (current_step_id_ < num_steps_ - 1) {
    const int64_t send_regst_desc_id = send_regst_desc_id_.at(current_ring_id_);
    Regst* send_regst = produced_rs_.Front(send_regst_desc_id);
    send_regst->set_piece_id(send_regst_piece_id_.at(current_ring_id_));
    send_regst_piece_id_[current_ring_id_] += 1;
    HandleRegstToConsumer(send_regst, [](int64_t) { return true; });
    produced_rs_.PopFrontRegsts({send_regst_desc_id});
  } else if (current_step_id_ == num_steps_ - 1) {
  } else {
    UNIMPLEMENTED();
  }
}

void MultiRingAllReduceActor::AsyncSendCustomizedConsumedRegstMsgToProducer() {
  if (current_step_id_ == num_steps_ - 1 && current_ring_id_ == num_rings_ - 1) {
    Regst* cur_regst = consumed_rs_.Front(in_regst_desc_id_);
    CHECK(cur_regst);
    AsyncSendRegstMsgToProducer(cur_regst);
    CHECK_EQ(0, consumed_rs_.TryPopFrontRegst(in_regst_desc_id_));
  }
  if (current_step_id_ > 0 && current_step_id_ <= num_steps_ - 1) {
    Regst* cur_regst = consumed_rs_.Front(recv_regst_desc_id_.at(current_ring_id_));
    CHECK(cur_regst);
    AsyncSendRegstMsgToProducer(cur_regst);
    CHECK_EQ(0, consumed_rs_.TryPopFrontRegst(recv_regst_desc_id_.at(current_ring_id_)));
  } else if (current_step_id_ == 0) {
  } else {
    UNIMPLEMENTED();
  }
  current_ring_id_ = (current_ring_id_ + 1) % num_rings_;
  if (current_ring_id_ == 0) { current_step_id_ = (current_step_id_ + 1) % num_steps_; }
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
  other_ctx_.second = current_step_id_;
  *other = &other_ctx_;
}

void MultiRingAllReduceActor::AsyncReturnAllCustomizedReadableRegst() {
  CHECK_EQ(0, consumed_rs_.available_regst_desc_cnt());
}

bool MultiRingAllReduceActor::ProducedCtrlRegstValid(int64_t regst_desc_id) const { return true; }

REGISTER_ACTOR(kMultiRingAllReduce, MultiRingAllReduceActor);

}  // namespace oneflow
