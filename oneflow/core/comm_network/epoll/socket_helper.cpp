#include "oneflow/core/comm_network/epoll/socket_helper.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

SocketHelper::SocketHelper(int sockfd, SocketIOWorker* read_worker,
                           SocketIOWorker* write_worker) {
  sockfd_ = sockfd;
  int opt = fcntl(sockfd_, F_GETFL);
  PCHECK(opt != -1);
  PCHECK(fcntl(sockfd_, F_SETFL, opt | O_NONBLOCK) == 0);
  read_helper_.reset(new SocketReadHelper(sockfd_, read_worker));
  write_helper_.reset(new SocketWriteHelper(sockfd_, write_worker));
}

}  // namespace oneflow

#endif  // PLATFORM_POSIX
