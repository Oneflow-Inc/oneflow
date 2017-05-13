#ifndef ONEFLOW_ACTOR_MESSAGE_H_
#define ONEFLOW_ACTOR_MESSAGE_H_

#include <stdint.h>

namespace oneflow {

struct Message {
  uint64_t dst_actor_id;
  uint64_t register_id;
};

}  // namespace oneflow

#endif  // ONEFLOW_ACTOR_MESSAGE_H_
