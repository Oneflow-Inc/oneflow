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
    std::vector<ActorMsg> actor_msgs_;
    Regst* current_in_regst = in_regst_deque_.front();
    if (step_conf.send()) {
      Regst* send = send_regst_.at(current_ring_id_);
      send->set_piece_id(send_regst_piece_id_.at(current_ring_id_));
      send_regst_piece_id_[current_ring_id_] += 1;
      for (const int64_t consumer : send_regst_.at(current_ring_id_)->consumers_actor_id()) {
        actor_msgs_.push_back(ActorMsg::BuildRegstMsgToConsumer(actor_id(), consumer, send));
      }
    }
    if (step_conf.recv()) {
      Regst* recv = recv_regst_.at(current_ring_id_);
      actor_msgs_.push_back(
          ActorMsg::BuildRegstMsgToProducer(actor_id(), recv->producer_actor_id(), recv));
    }
    if (current_step_id_ == num_steps_ - 1 && current_ring_id_ == num_rings_ - 1) {
      out_regst_->set_piece_id(current_in_regst->piece_id());
      for (const int64_t consumer : out_regst_->consumers_actor_id()) {
        actor_msgs_.push_back(ActorMsg::BuildRegstMsgToConsumer(actor_id(), consumer, out_regst_));
      }
      out_regst_desc_reading_cnt_ = out_regst_->consumers_actor_id().size();
      actor_msgs_.push_back(ActorMsg::BuildRegstMsgToProducer(
          actor_id(), current_in_regst->producer_actor_id(), current_in_regst));
      in_regst_deque_.pop_front();
    }
    AsyncDo([actor_msgs_]() {
      for (const auto& msg : actor_msgs_) { Global<ActorMsgBus>::Get()->SendMsg(msg); }
    });
    current_ring_id_ = (current_ring_id_ + 1) % num_rings_;
    if (current_ring_id_ == 0) { current_step_id_ = (current_step_id_ + 1) % num_steps_; }
  }
  if (in_regst_eord_ && in_regst_deque_.empty() && current_ring_id_ == 0 && current_step_id_ == 0
      && out_regst_desc_reading_cnt_ == 0) {
    for (const int64_t consumer : out_regst_->consumers_actor_id()) {
      std::vector<ActorMsg> actor_msgs_;
      actor_msgs_.push_back(ActorMsg::BuildEordMsg(consumer, out_regst_->regst_desc_id()));
      AsyncDo([actor_msgs_]() {
        for (const auto& msg : actor_msgs_) { Global<ActorMsgBus>::Get()->SendMsg(msg); }
      });
    }
    OF_SET_MSG_HANDLER(nullptr);
    return 1;
  }
  return 0;
}

void MultiRingAllReduceActor::VirtualActorInit(const TaskProto& task_proto) {
  CHECK_EQ(1, exec_kernel_vec().size());
  out_regst_desc_id_ = task_proto.produced_regst_desc().at("out").regst_desc_id();
  out_regst_ = GetSoleProducedRegst4RegstDescId(out_regst_desc_id_);
  out_regst_desc_reading_cnt_ = 0;
  CHECK_EQ(task_proto.consumed_regst_desc_id().at("in").regst_desc_id_size(), 1);
  in_regst_desc_id_ = task_proto.consumed_regst_desc_id().at("in").regst_desc_id(0);
  multi_ring_all_reduce_kernel_conf_ =
      task_proto.exec_sequence().exec_node(0).kernel_conf().multi_ring_all_reduce_conf();
  num_rings_ = multi_ring_all_reduce_kernel_conf_.ring_conf_size();
  CHECK_GE(num_rings_, 1);
  num_steps_ = multi_ring_all_reduce_kernel_conf_.ring_conf(0).step_conf_size();
  FOR_RANGE(int64_t, ring_id, 1, num_rings_) {
    CHECK_EQ(multi_ring_all_reduce_kernel_conf_.ring_conf(1).step_conf_size(), num_steps_);
  }
  FOR_RANGE(int64_t, ring_id, 0, num_rings_) {
    const std::string send_name = "send_" + std::to_string(ring_id);
    const std::string recv_name = "recv_" + std::to_string(ring_id);
    const int64_t send_regst_desc_id =
        task_proto.produced_regst_desc().at(send_name).regst_desc_id();
    send_regst_piece_id_.push_back(0);
    send_regst_.push_back(GetSoleProducedRegst4RegstDescId(send_regst_desc_id));
    send_regst_ready_.push_back(true);
    CHECK(regst_desc_id2send_or_recv7ring_id_
              .emplace(send_regst_desc_id, std::make_pair(true, ring_id))
              .second);
    CHECK_EQ(task_proto.consumed_regst_desc_id().at(recv_name).regst_desc_id_size(), 1);
    const int64_t recv_regst_desc_id =
        task_proto.consumed_regst_desc_id().at(recv_name).regst_desc_id(0);
    recv_regst_ready_.push_back(false);
    CHECK(regst_desc_id2send_or_recv7ring_id_
              .emplace(recv_regst_desc_id, std::make_pair(false, ring_id))
              .second);
  }
  in_regst_eord_ = false;
  current_ring_id_ = 0;
  current_step_id_ = 0;
  OF_SET_MSG_HANDLER(&MultiRingAllReduceActor::HandlerAllReduce);
}

REGISTER_ACTOR(kMultiRingAllReduce, MultiRingAllReduceActor);

}  // namespace oneflow
