#include "oneflow/core/actor/boxing_actor.h"
#include "oneflow/core/actor/actor_registry.h"
#include "oneflow/core/register/local_register_warpper.h"

namespace oneflow {

void BoxingActor::Init(const TaskProto& task_proto) {
  Actor::Init(task_proto);
  in_regst_desc_num_ = task_proto.subscribed_regst_desc_id().size();
}

void BoxingActor::ProcessMsg(const ActorMsg& msg) {
  if (TryOneReadDone(msg.regst_warpper()->regst_raw_ptr()) != 0) {
    std::shared_ptr<RegstWarpper> regst_wp = msg.regst_warpper();
    auto waiting_in_regst_it = waiting_in_regst_.find(regst_wp->piece_id());
    if (waiting_in_regst_it == waiting_in_regst_.end()) {
      auto emplace_ret = waiting_in_regst_.emplace(
          regst_wp->piece_id(), of_make_unique<RDescId2RwMap> ());
      CHECK(emplace_ret.second);
      waiting_in_regst_it = emplace_ret.first;
    }
    CHECK(waiting_in_regst_it->second->emplace(regst_wp->regst_desc_id(),
                                               regst_wp).second);
    if (waiting_in_regst_it->second->size() == in_regst_desc_num_) {
      std::pair<uint64_t, RDescId2RwMapPtr> ready_ins;
      ready_ins.first = waiting_in_regst_it->first;
      ready_ins.second.swap(waiting_in_regst_it->second);
      ready_in_regst_.push(std::move(ready_ins));
      waiting_in_regst_.erase(waiting_in_regst_it);
    }
  }
  if (!ready_in_regst_.empty() && IsWriteReady()) {
    WardKernelAndSendMsg();
  }
}

void BoxingActor::WardKernelAndSendMsg() {
  uint64_t piece_id = ready_in_regst_.front().first;
  WardKernel([this](uint64_t regst_desc_id) -> std::shared_ptr<RegstWarpper> {
    Regst* regst = GetCurWriteableRegst(regst_desc_id);
    if (regst == nullptr) {
      return ready_in_regst_.front().second->at(regst_desc_id);
    } else {
      return std::make_shared<LocalRegstWarpper> (regst);
    }
  });
  ForEachCurWriteableRegst([piece_id](Regst* regst) {
    regst->set_piece_id(piece_id);
  });
  CurWriteDone();
  for (const auto& pair : *(ready_in_regst_.front().second)) {
    std::shared_ptr<RegstWarpper> regst = pair.second;
    ActorMsgBus::Singleton().SendMsg(ActorMsg::BuildMsgForRegstWriter(
        regst->producer_actor_id(),
        regst->regst_raw_ptr()));
  }
  ready_in_regst_.pop();
}

REGISTER_ACTOR(kBoxingTask, true, BoxingActor);
REGISTER_ACTOR(kBoxingTask, false, BoxingActor);

}  // namespace oneflow
