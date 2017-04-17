#ifndef THOR_NETWORK_MESSAGE_H_
#define THOR_NETWORK_MESSAGE_H_
#include <cstdint>
#include "thread/event_message.h"

namespace oneflow {

enum class NetworkMessageType {
  MSG_TYPE_BARRIER = 1,
  MSG_TYPE_REPLY_BARRIER = -1,

  MSG_TYPE_REMOTE_MEMORY_DESCRIPTOR = 2,

  MSG_TYPE_REQUEST_ACK = 3
};

struct NetworkMessage {
  // required
  NetworkMessageType type;
  int32_t src_rank;
  int32_t dst_rank;

  // Request/ack EventMessage between send/recv actors for MSG_TYPE_READ_OK or
  // MSG_TYPE_ACK_CONSUMED
  // Also, includes the Write completion event message to send actor
  EventMessage event_msg;

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

#endif  // THOR_NETWORK_MESSAGE_H_
