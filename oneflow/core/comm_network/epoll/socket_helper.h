#ifndef ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_HELPER_H_
#define ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_HELPER_H_

#include "oneflow/core/comm_network/epoll/io_event_poller.h"
#include "oneflow/core/comm_network/epoll/socket_read_helper.h"
#include "oneflow/core/comm_network/epoll/socket_write_helper.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

class SocketHelper final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SocketHelper);
  SocketHelper() = delete;
  ~SocketHelper();

  SocketHelper(int sockfd, IOEventPoller* poller);

  void AsyncWrite(const SocketMsg& msg);

 private:
  SocketReadHelper* read_helper_;
  SocketWriteHelper* write_helper_;
};

}  // namespace oneflow

#endif  // PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_HELPER_H_
