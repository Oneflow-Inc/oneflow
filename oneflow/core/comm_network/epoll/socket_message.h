#ifndef ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_MESSAGE_H_
#define ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_MESSAGE_H_

#include "oneflow/core/common/platform.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/comm_network/comm_network.h"

#ifdef PLATFORM_POSIX

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include "oneflow/core/actor/actor_message.h"

namespace oneflow {

#define SOCKET_MSG_TYPE_SEQ                         \
  OF_PP_MAKE_TUPLE_SEQ(RequestWrite, request_write) \
  OF_PP_MAKE_TUPLE_SEQ(RequestRead, request_read)   \
  OF_PP_MAKE_TUPLE_SEQ(Actor, actor)

enum class SocketMsgType {
#define MAKE_ENTRY(x, y) k##x,
  OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, SOCKET_MSG_TYPE_SEQ)
#undef MAKE_ENTRY
};

struct RequestWriteMsg {
  void* src_token;
  int64_t dst_machine_id;
  void* dst_token;
  int64_t stream_id;
};

struct RequestReadMsg {
  void* src_token;
  void* dst_token;
  int64_t offset;
  int64_t byte_size;
  int64_t stream_id;
  int32_t part_num;
};

struct SocketMsg {
  SocketMsgType msg_type;
  union {
#define MAKE_ENTRY(x, y) x##Msg y##_msg;
    OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, SOCKET_MSG_TYPE_SEQ)
#undef MAKE_ENTRY
  };
};

using CallBackList = std::list<std::function<void()>>;

}  // namespace oneflow

#endif  // PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_MESSAGE_H_
