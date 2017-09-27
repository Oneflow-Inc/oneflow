#ifndef ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_WRITE_HELPER_H_
#define ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_WRITE_HELPER_H_

#include "oneflow/core/device/cpu_stream.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

class SocketWriteHelper final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SocketWriteHelper);
  SocketWriteHelper() = delete;
  ~SocketWriteHelper() { TODO(); }

  SocketWriteHelper(int sockfd, CpuStream* cpu_stream) { TODO(); }

  void AsyncWrite(const SocketMsg& msg) { TODO(); }

  void NotifyMeSocketWriteable() { TODO(); }

 private:
};

}  // namespace oneflow

#endif  // PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_WRITE_HELPER_H_
