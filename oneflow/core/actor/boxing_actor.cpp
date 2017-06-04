#include "oneflow/core/actor/boxing_actor.h"
#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

void BoxingActor::Init(const TaskProto& task_proto) {
  Actor::Init(task_proto);
  in_regst_desc_num_ = task_proto.subscribed_regst_desc_id().size();
  uint64_t middle_regst_desc_id = GetRegstDescIdFromName("middle");
  for (const auto& regst : produced_regst_vec()) {
    if (regst->regst_desc_id() == middle_regst_desc_id) {
      LOG_IF(WARNING, middle_regst_ != nullptr) << "redundant middle register";
      middle_regst_ = regst.get();
    } else {
      writeable_out_regst_[regst->regst_desc_id()].push(regst.get());
      out_regst2reading_cnt_[regst.get()] = 0;
    }
  }
  writeable_out_regst_desc_num_ = writeable_out_regst_.size();
}

void BoxingActor::ProcessMsg(const ActorMsg& msg) {
  auto writeable_out_regst_it = writeable_out_regst_.find(msg.regst()->regst_desc_id());
  if (writeable_out_regst_it != writeable_out_regst_.end()) {
    auto reading_cnt_it = out_regst2reading_cnt_.find(msg.regst());
    CHECK_GE(reading_cnt_it->second, 1);
    reading_cnt_it->second -= 1;
    if (reading_cnt_it->second == 0) {
      if (writeable_out_regst_it->second.empty()) {
        writeable_out_regst_desc_num_ += 1;
      }
      writeable_out_regst_it->second.push(msg.regst());
    }
  } else {
    auto waiting_in_regst_it = waiting_in_regst_.find(msg.piece_id());
    if (waiting_in_regst_it == waiting_in_regst_.end()) {
      auto emplace_ret = waiting_in_regst_.emplace(
          msg.piece_id(), of_make_unique<HashMap<uint64_t, Regst*>> ());
      CHECK(emplace_ret.second);
      waiting_in_regst_it = emplace_ret.first;
    }
    CHECK(waiting_in_regst_it->second->emplace(msg.regst()->regst_desc_id(),
                                               msg.regst()).second);
    if (waiting_in_regst_it->second->size() == in_regst_desc_num_) {
      std::pair<uint64_t, std::unique_ptr<HashMap<uint64_t, Regst*>>> ready_ins;
      ready_ins.first = waiting_in_regst_it->first;
      ready_ins.second.swap(waiting_in_regst_it->second);
      ready_in_regst_.push(std::move(ready_ins));
      waiting_in_regst_.erase(waiting_in_regst_it);
    }
  }
  if (!ready_in_regst_.empty()
      && writeable_out_regst_desc_num_ == writeable_out_regst_.size()) {
    WardKernelAndSendMsg();
  }
}

void BoxingActor::WardKernelAndSendMsg() {
  // Ward Kernel
  WardKernel([this](uint64_t regst_desc_id) {
    if (regst_desc_id == middle_regst_->regst_desc_id()) {
      return middle_regst_;
    }
    auto writeable_out_regst_it = writeable_out_regst_.find(regst_desc_id);
    if (writeable_out_regst_it != writeable_out_regst_.end()) {
      return writeable_out_regst_it->second.front();
    } else {
      return ready_in_regst_.front().second->at(regst_desc_id);
    }
  });
  // Send Msg about Out Regst
  {
    ActorMsg msg;
    msg.set_piece_id(ready_in_regst_.front().first);
    for (auto& pair : writeable_out_regst_) {
      Regst* regst = pair.second.front();
      out_regst2reading_cnt_.at(regst) = regst->subscribers_actor_id().size();
      msg.set_regst(regst);
      for (uint64_t subscriber : regst->subscribers_actor_id()) {
        msg.set_dst_actor_id(subscriber);
        ActorMsgBus::Singleton().SendMsg(msg);
      }
      pair.second.pop();
      if (pair.second.empty()) { writeable_out_regst_desc_num_ -= 1; }
    }
  }
  // Send Msg about In Regst
  for (const auto& pair : *(ready_in_regst_.front().second)) {
    Regst* regst = pair.second;
    ActorMsg msg;
    msg.set_dst_actor_id(regst->producer_actor_id());
    msg.set_regst(regst);
    ActorMsgBus::Singleton().SendMsg(msg);
  }
  ready_in_regst_.pop();
}

REGISTER_ACTOR(kBoxingTask, true, BoxingActor);
REGISTER_ACTOR(kBoxingTask, false, BoxingActor);

}  // namespace oneflow
