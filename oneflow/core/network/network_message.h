#ifndef ONEFLOW_CORE_NETWORK_NETWORK_MESSAGE_H_
#define ONEFLOW_CORE_NETWORK_NETWORK_MESSAGE_H_
#include "oneflow/core/actor/actor_message.h"

namespace oneflow {

enum NetworkMessageType {
  kBarrier = 0,
  kReplyBarrier = 1,
  kRemoteMemoryDescriptor = 2,
  kRequestAck = 3
};

struct NetworkMessage {
  // required
  NetworkMessageType type;
  int64_t src_machine_id;
  int64_t dst_machine_id;

  // Request/ack ActorMessage between send/recv actors for MSG_TYPE_READ_OK or
  // MSG_TYPE_ACK_CONSUMED
  ActorMsg actor_msg;

  // Optional for REMOTE_MEMORY_DESCRIPTOR message
  uint64_t address;
  uint32_t token;
};

enum NetworkResultType {
  kReadOk = 0,
  kSendOk = 1,
  kReceiveMsg = 2
};

struct NetworkResult {
  NetworkResultType type;
  // Used when type == kReceiveMsg, or type == kReadOk, msg.
  NetworkMessage net_msg;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORK_NETWORK_MESSAGE_H_
