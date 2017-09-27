#ifndef ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_READ_HELPER_H_
#define ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_READ_HELPER_H_

#include "oneflow/core/device/cpu_stream.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

class SocketReadHelper final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SocketReadHelper);
  SocketReadHelper() = delete;
  ~SocketReadHelper() { TODO(); }

  SocketReadHelper(int sockfd, CpuStream* cpu_stream) { TODO(); }

  void NotifyMeSocketReadable() { TODO(); }

 private:
};

}  // namespace oneflow

#endif  // PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_READ_HELPER_H_
