#include "oneflow/core/comm_network/epoll/socket_helper.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

SocketHelper::SocketHelper(int sockfd, IOEventPoller* poller,
                           CpuStream* read_cpu_stream,
                           CpuStream* write_cpu_stream) {
  read_helper_ = new SocketReadHelper(sockfd, read_cpu_stream);
  write_helper_ = new SocketWriteHelper(sockfd, write_cpu_stream);
  poller->AddFd(sockfd, [this]() { read_helper_->NotifyMeSocketReadable(); },
                [this]() { write_helper_->NotifyMeSocketWriteable(); });
}

void SocketHelper::AsyncWrite(const SocketMsg& msg) {
  write_helper_->AsyncWrite(msg);
}

}  // namespace oneflow

#endif  // PLATFORM_POSIX
