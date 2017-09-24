#ifndef ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_IO_WORKER_H_
#define ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_IO_WORKER_H_

#include "oneflow/core/comm_network/epoll/socket_message.h"
#include "oneflow/core/common/channel.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

class SocketIOHelperIf {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SocketIOHelperIf);
  virtual ~SocketIOHelperIf() = default;

  virtual void Work() = 0;

 protected:
  SocketIOHelperIf() = default;
};

class SocketIOWorker final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SocketIOWorker);
  SocketIOWorker();
  ~SocketIOWorker();

  void ProcessReadyIOHelper(SocketIOHelperIf* io_helper);

 private:
  std::thread thread_;
  Channel<SocketIOHelperIf*> channel_;
};

}  // namespace oneflow

#endif  // PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_IO_WORKER_H_
