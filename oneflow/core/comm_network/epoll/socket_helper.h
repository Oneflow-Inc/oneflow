#ifndef ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_HELPER_H_
#define ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_HELPER_H_

#include "oneflow/core/comm_network/epoll/socket_read_helper.h"
#include "oneflow/core/comm_network/epoll/socket_write_helper.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

class SocketHelper final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SocketHelper);
  SocketHelper() = delete;
  ~SocketHelper() = default;

  SocketHelper(int sockfd, SocketIOWorker* read_worker,
               SocketIOWorker* write_worker);

  SocketReadHelper* mut_read_helper() { return read_helper_.get(); }
  SocketWriteHelper* mut_write_helper() { return write_helper_.get(); }

 private:
  int sockfd_;
  std::unique_ptr<SocketReadHelper> read_helper_;
  std::unique_ptr<SocketWriteHelper> write_helper_;
};

}  // namespace oneflow

#endif  // PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_HELPER_H_
