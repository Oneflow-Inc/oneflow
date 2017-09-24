#include "oneflow/core/comm_network/epoll/socket_io_worker.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

SocketIOWorker::SocketIOWorker() {
  thread_ = std::thread([this]() {
    SocketIOHelperIf* io_helper;
    while (channel_.Receive(&io_helper) == 0) { io_helper->Work(); }
  });
}

SocketIOWorker::~SocketIOWorker() {
  channel_.CloseSendEnd();
  thread_.join();
  channel_.CloseReceiveEnd();
}

void AddWork(SocketIOHelperIf* io_helper) {
  CHECK_EQ(channel_.Send(io_helper), 0);
}

}  // namespace oneflow

#endif  // PLATFORM_POSIX
