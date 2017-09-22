#ifndef ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_IO_HELPER_H_
#define ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_IO_HELPER_H_

#include "oneflow/core/common/platform.h"
#include "oneflow/core/common/util.h"

#ifdef PLATFORM_POSIX

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>

#include <fcntl.h>
#include <unistd.h>

namespace oneflow {

class SocketIOHelper final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SocketIOHelper);
  SocketIOHelper() = delete;
  ~SocketIOHelper() = default;

  SocketIOHelper(int sockfd);

 private:
  int sockfd_;
};

}  // namespace oneflow

#endif  // PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_IO_HELPER_H_
