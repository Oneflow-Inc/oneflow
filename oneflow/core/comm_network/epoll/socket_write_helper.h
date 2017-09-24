#ifndef ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_WRITE_HELPER_H_
#define ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_WRITE_HELPER_H_

#include "oneflow/core/comm_network/epoll/socket_io_worker.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

class SocketWriteHelper final : public SocketIOHelperIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SocketWriteHelper);
  SocketWriteHelper() = delete;
  ~SocketWriteHelper() { TODO(); }

  SocketWriteHelper(int sockfd, SocketIOWorker* worker) { TODO(); }

  void Write(const SocketMsg& msg) { TODO(); }

  void Work() override { TODO(); }

  void NotifyWorker() { TODO(); }

 private:
};

}  // namespace oneflow

#endif  // PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_WRITE_HELPER_H_
