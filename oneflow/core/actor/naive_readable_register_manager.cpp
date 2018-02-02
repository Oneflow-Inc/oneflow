#include "oneflow/core/actor/naive_readable_register_manager.h"

namespace oneflow {

void NaiveReadableRegstMgr::Init(const TaskProto& task_proto) {
  for (const auto& pair : task_proto.consumed_regst_desc_id()) {
    readable_regst_[pair.second] = {};
  }
  readable_regst_cnt_ = 0;
}

void NaiveReadableRegstMgr::Push(Regst* regst) {
  std::queue<Regst*>& rq = readable_regst_.at(regst->regst_desc_id());
  if (rq.empty()) { readable_regst_cnt_ += 1; }
  rq.push(regst);
}

void NaiveReadableRegstMgr::ReturnToProducerAndPopCurReadable(
    Actor* actor, std::function<bool(Regst*)> IsAllowed) {
  for (auto& pair : readable_regst_) {
    CHECK_EQ(pair.second.empty(), false);
    if (IsAllowed(pair.second.front()) == false) { continue; }
    actor->AsyncSendRegstMsgToProducer(pair.second.front());
    pair.second.pop();
    if (pair.second.empty()) { readable_regst_cnt_ -= 1; }
  }
}

void NaiveReadableRegstMgr::ReturnToProducerAndPopCurReadable(Actor* actor) {
  ReturnToProducerAndPopCurReadable(actor, [](Regst*) { return true; });
}

Regst* NaiveReadableRegstMgr::GetCurReadable(int64_t regst_desc_id) {
  auto it = readable_regst_.find(regst_desc_id);
  if (it != readable_regst_.end() && it->second.empty() == false) {
    return it->second.front();
  } else {
    return nullptr;
  }
}

void NaiveReadableRegstMgr::ForEachCurReadableRegst(
    std::function<void(Regst*)> func) {
  for (const auto& pair : readable_regst_) {
    if (pair.second.empty() == false) { func(pair.second.front()); }
  }
}

bool NaiveReadableRegstMgr::IsReadReady() {
  return readable_regst_.size() == readable_regst_cnt_;
}

}  // namespace oneflow
