#include "oneflow/core/comm_network/epoll/socket_io_helper.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

SocketIOHelper::SocketIOHelper(int sockfd) {
  sockfd_ = sockfd;
  int opt = fcntl(sockfd_, F_GETFL);
  PCHECK(opt != -1);
  PCHECK(fcntl(sockfd_, F_SETFL, opt | O_NONBLOCK) == 0);
}

}  // namespace oneflow

#endif  // PLATFORM_POSIX
