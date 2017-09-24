#ifndef ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_MESSAGE_H_
#define ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_MESSAGE_H_

#include "oneflow/core/common/platform.h"
#include "oneflow/core/common/util.h"

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

enum class SocketMsgType { kRequestWrite = 0, kActor };

struct RequestWriteMsg {
  const void* src_token;
  int64_t dst_machine_id;
  const void* dst_token;
  void* read_id;
};

struct SocketMsg {
  SocketMsgType msg_type;
  union {
    RequestWriteMsg request_write_msg;
    ActorMsg actor_msg;
  };
};

using CallBackList = std::list<std::function<void()>>;

}  // namespace oneflow

#endif  // PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_MESSAGE_H_
