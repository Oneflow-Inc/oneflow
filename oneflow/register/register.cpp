#include "register/register.h"

namespace oneflow {

void Regst::ProduceDone() {
  CHECK_EQ(cnt_.load(), 0);
  cnt_.store(consumer_ids_.size());
  ActorMsg m;
  m.register_id = id_;
  for (uint64_t consumer_id : consumer_ids_) {
    m.to_actor_id = consumer_id;
    ActorMsgBus::Singleton().SendMsg(m);
  }
}

void Regst::ConsumeDone() {
  --cnt_;
  if (cnt_.load() == 0) {
    ActorMsg m;
    m.register_id = id_;
    m.to_actor_id = producer_id_;
    ActorMsgBus::Singleton().SendMsg(m);
  }
}

Blob* Regst::GetBlobFromLbn(const std::string& lbn) {
  return lbn2blob_.at(lbn).get();
}

}
