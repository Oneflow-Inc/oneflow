#ifndef ONEFLOW_NETWORK_NETWORK_MESSAGE_H_
#define ONEFLOW_NETWORK_NETWORK_MESSAGE_H_
#include <cstdint>
//#include "runtime/event_message.h"
#include "actor/actor_message.h"

namespace oneflow {
typedef std::shared_ptr<ActorMsg> MsgPtr;


enum class NetworkMessageType {
  MSG_TYPE_BARRIER = 1,
  MSG_TYPE_REPLY_BARRIER = -1,

  MSG_TYPE_REMOTE_MEMORY_DESCRIPTOR = 2,

  MSG_TYPE_REQUEST_ACK = 3
};

struct NetworkMessage {
  // required
  NetworkMessageType type;
  uint64_t src_machine_id;
  uint64_t dst_machine_id;

  // Request/ack EventMessage between send/recv actors for MSG_TYPE_READ_OK or
  // MSG_TYPE_ACK_CONSUMED
  // Also, includes the Write completion event message to send actor
  ActorMsg actor_msg;

  // Optional for REMOTE_MEMORY_DESCRIPTOR message
  uint64_t address;
  uint32_t token;
};

enum class NetworkResultType {
  NET_READ_OK,
  NET_SEND_OK,
  NET_RECEIVE_MSG
};

struct NetworkResult {
  NetworkResultType type;
  // Used when type == NET_RECEIVE_MSG, or type == NET_READ_OK, msg.
  NetworkMessage net_msg;
};

}  // namespace oneflow

#endif  // ONEFLOW_NETWORK_NETWORK_MESSAGE_H_
