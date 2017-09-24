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

namespace oneflow {

struct SocketMsg {
  // TODO
};

}  // namespace oneflow

#endif  // PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_MESSAGE_H_
