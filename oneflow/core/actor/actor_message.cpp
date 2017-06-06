#include "oneflow/core/actor/actor_message.h"

namespace oneflow {

ActorMsg::ActorMsg() {
  dst_actor_id_ = std::numeric_limits<uint64_t>::max();
}

ActorMsg ActorMsg::BuildMsgForRegstReader(uint64_t reader_actor_id,
                                          Regst* regst_raw_ptr) {
  TODO();
}

ActorMsg ActorMsg::BuildMsgForRegstWriter(uint64_t writer_actor_id,
                                          Regst* regst_raw_ptr) {
  TODO();
}

} // namespace oneflow
