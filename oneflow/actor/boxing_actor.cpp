#include "actor/boxing_actor.h"
#include "actor/actor_registry.h"

namespace oneflow {

void BoxingActor::Init(const TaskProto& task_proto) {
  Actor::Init(task_proto);
  subscribed_regst_desc_num_ = task_proto.subscribed_regst_desc_id().size();
  for (const auto& regst : produced_regst_vec()) {
    waiting_pregst_[regst->regst_desc_id()].push(regst.get());
  }
  waiting_pregst_desc_num_ = waiting_pregst_.size();
}

void BoxingActor::ProcessMsg(const ActorMsg& msg) {
  auto waiting_pregst_it = waiting_pregst_.find(msg.regst()->regst_desc_id());
  if (waiting_pregst_it != waiting_pregst_.end()) {
    if (waiting_pregst_it->second.empty()) { waiting_pregst_desc_num_ += 1; }
    waiting_pregst_it->second.push(msg.regst());
  } else {
    auto waiting_sregst_it = waiting_sregst_.find(msg.piece_id());
    if (waiting_sregst_it == waiting_sregst_.end()) {
      auto emplace_ret = waiting_sregst_.emplace(
          msg.piece_id(), of_make_unique<HashMap<uint64_t, Regst*>> ());
      CHECK(emplace_ret.second);
      waiting_sregst_it = emplace_ret.first;
    }
    CHECK(waiting_sregst_it->second->emplace(msg.regst()->regst_desc_id(),
                                             msg.regst()).second);
    if (waiting_sregst_it->second->size() == subscribed_regst_desc_num_) {
      std::pair<uint64_t, std::unique_ptr<HashMap<uint64_t, Regst*>>> ready_ins;
      ready_ins.first = waiting_sregst_it->first;
      ready_ins.second.swap(waiting_sregst_it->second);
      ready_sregst_.push(std::move(ready_ins));
      waiting_sregst_.erase(waiting_sregst_it);
    }
  }
  if (!ready_sregst_.empty()
      && waiting_pregst_desc_num_ == waiting_pregst_.size()) {
    WardKernelAndSendMsg();
  }
}

void BoxingActor::WardKernelAndSendMsg() {
  // Ward Kernel
  WardKernel([this](uint64_t regst_desc_id) {
    auto waiting_pregst_it = waiting_pregst_.find(regst_desc_id);
    if (waiting_pregst_it != waiting_pregst_.end()) {
      return waiting_pregst_it->second.front();
    } else {
      return ready_sregst_.front().second->at(regst_desc_id);
    }
  });
  // Send Msg about Produced Regst
  {
    ActorMsg msg;
    msg.set_piece_id(ready_sregst_.front().first);
    for (auto& pair : waiting_pregst_) {
      Regst* regst = pair.second.front();
      msg.set_regst(regst);
      for (uint64_t subscriber : regst->subscribers_actor_id()) {
        msg.set_dst_actor_id(subscriber);
        ActorMsgBus::Singleton().SendMsg(msg);
      }
      pair.second.pop();
      if (pair.second.empty()) { waiting_pregst_desc_num_ -= 1; }
    }
  }
  // Send Msg about Subscribed Regst
  for (const auto& pair : *(ready_sregst_.front().second)) {
    Regst* regst = pair.second;
    ActorMsg msg;
    msg.set_dst_actor_id(regst->producer_actor_id());
    msg.set_regst(regst);
    ActorMsgBus::Singleton().SendMsg(msg);
  }
  ready_sregst_.pop();
}

REGISTER_ACTOR(kBoxingTask, true, BoxingActor);
REGISTER_ACTOR(kBoxingTask, false, BoxingActor);

}  // namespace oneflow
