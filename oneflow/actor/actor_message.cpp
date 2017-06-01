#include "actor/actor_message.h"

namespace oneflow {

ActorMsg::ActorMsg() {
  dst_actor_id_ = std::numeric_limits<uint64_t>::max();
  piece_id_ = std::numeric_limits<uint64_t>::max();
  regst_ = nullptr;
  regst_dptr_ = nullptr;
}

} // namespace oneflow
