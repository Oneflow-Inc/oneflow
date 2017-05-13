#include "register/register.h"

namespace oneflow {

void Regst::ProduceDone() {
  ++cnt_;
  CHECK_EQ(cnt_, 0);
  cnt = consumer_ids_.size();
  Message m;
  m.register_id = id_;
  for (uint64_t consumer_id : consumer_ids_) {
    m.to_actor_id = consumer_id;
    Commbus::Singleton().SendMsg(m);
  }
}

void Regst::ConsumeDone() {
  --cnt_;
  if (cnt_ == 0) {
    Message m;
    m.register_id = id_;
    m.to_actor_id = producer_id_;
    Commbus::Singleton().SendMsg(m);
  }
}

Blob* Regst::GetBlobFromLbn(const std::string& lbn) {
  return lbn2blob_.at(lbn).get();
}

}
