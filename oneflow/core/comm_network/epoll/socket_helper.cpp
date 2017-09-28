#include "oneflow/core/comm_network/epoll/socket_helper.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

SocketHelper::SocketHelper(int sockfd, IOEventPoller* poller) {
  read_helper_ = new SocketReadHelper(sockfd);
  write_helper_ = new SocketWriteHelper(sockfd, poller);
  poller->AddFd(sockfd, [this]() { read_helper_->NotifyMeSocketReadable(); },
                [this]() { write_helper_->NotifyMeSocketWriteable(); });
}

SocketHelper::~SocketHelper() {
  delete read_helper_;
  delete write_helper_;
}

void SocketHelper::AsyncWrite(const SocketMsg& msg) {
  write_helper_->AsyncWrite(msg);
}

}  // namespace oneflow

#endif  // PLATFORM_POSIX
