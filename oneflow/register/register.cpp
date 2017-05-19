#include "register/register.h"

namespace oneflow {

void Regst::ProduceDone() {
  std::unique_lock<std::mutex> lock(mtx_);
  CHECK_EQ(cnt_, 0);
  cnt_ = consumer_ids_.size();
  ActorMsg m;
  m.set_register_id(id_);
  for (uint64_t consumer_id : consumer_ids_) {
    m.set_dst_actor_id(consumer_id);
    ActorMsgBus::Singleton().SendMsg(m);
  }
}

void Regst::ConsumeDone() {
  std::unique_lock<std::mutex> lock(mtx_);
  --cnt_;
  if (cnt_ == 0) {
    ActorMsg m;
    m.set_register_id(id_);
    m.set_dst_actor_id(producer_id_);
    ActorMsgBus::Singleton().SendMsg(m);
  }
}

Blob* Regst::GetBlobPtrFromLbn(const std::string& lbn) {
  return lbn2blob_.at(lbn).get();
}

}
