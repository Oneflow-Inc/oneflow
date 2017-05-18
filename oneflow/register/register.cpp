#include "register/register.h"

namespace oneflow {

void Regst::ProduceDone() {
  std::unique_lock<std::mutex> lock(mtx_);
  CHECK_EQ(cnt_, 0);
  cnt_ = consumer_ids_.size();
  lock.unlock();
  ActorMsg m;
  m.register_id = id_;
  for (uint64_t consumer_id : consumer_ids_) {
    m.to_actor_id = consumer_id;
    ActorMsgBus::Singleton().SendMsg(m);
  }
}

void Regst::ConsumeDone() {
  std::unique_lock<std::mutex> lock(mtx_);
  --cnt_;
  if (cnt_ == 0) {
    ActorMsg m;
    m.register_id = id_;
    m.to_actor_id = producer_id_;
    ActorMsgBus::Singleton().SendMsg(m);
  }
  lock.unlock();
}

Blob* Regst::GetBlobPtrFromLbn(const std::string& lbn) {
  return lbn2blob_.at(lbn).get();
}

}
