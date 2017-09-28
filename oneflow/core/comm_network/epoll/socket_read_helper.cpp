#include "oneflow/core/comm_network/epoll/socket_read_helper.h"
#include "oneflow/core/comm_network/epoll/epoll_data_comm_network.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

SocketReadHelper::~SocketReadHelper() { TODO(); }

SocketReadHelper::SocketReadHelper(int sockfd, CpuStream* cpu_stream) {
  sockfd_ = sockfd;
  cpu_stream_ = cpu_stream;
  cur_read_handle_ = &SocketReadHelper::MsgHeadReadHandle;
  read_ptr_ = reinterpret_cast<char*>(&cur_msg_);
  read_size_ = sizeof(cur_msg_);
}

void SocketReadHelper::NotifyMeSocketReadable() {
  cpu_stream_->SendWork(
      std::bind(&SocketReadHelper::ReadUntilSocketNotReadable, this));
}

void SocketReadHelper::ReadUntilSocketNotReadable() {
  while ((this->*cur_read_handle_)()) {}
}

bool SocketReadHelper::MsgHeadReadHandle() {
  DoCurRead(&SocketReadHelper::SetStatusWhenMsgHeadDone);
}

bool SocketReadHelper::MsgBodyReadHandle() {
  DoCurRead(&SocketReadHelper::SetStatusWhenMsgBodyDone);
}

bool SocketReadHelper::DoCurRead(
    bool (SocketReadHelper::*set_cur_read_done)()) {
  ssize_t n = read(sockfd_, read_ptr_, read_size_);
  if (n == read_size_) {
    return (this->*set_cur_read_done)();
  } else if (n >= 0) {
    read_ptr_ += n;
    read_size_ -= n;
    return true;
  } else {
    CHECK_EQ(n, -1);
    PCHECK(errno == EAGAIN || errno == EWOULDBLOCK);
    return false;
  }
}

bool SocketReadHelper::SetStatusWhenMsgHeadDone() {
  switch (cur_msg_.msg_type) {
#define MAKE_ENTRY(x, y) \
  case SocketMsgType::k##x: return SetStatusWhen##x##MsgHeadDone();
    OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, SOCKET_MSG_TYPE_SEQ);
#undef MAKE_ENTRY
    default: UNEXPECTED_RUN();
  }
  UNEXPECTED_RUN();
}

bool SocketReadHelper::SetStatusWhenMsgBodyDone() {
  cur_read_handle_ = &SocketReadHelper::MsgHeadReadHandle;
  read_ptr_ = reinterpret_cast<char*>(&cur_msg_);
  read_size_ = sizeof(cur_msg_);
  return true;
}

bool SocketReadHelper::SetStatusWhenRequestWriteMsgHeadDone() { TODO(); }

bool SocketReadHelper::SetStatusWhenRequestReadMsgHeadDone() { TODO(); }

bool SocketReadHelper::SetStatusWhenActorMsgHeadDone() { TODO(); }

}  // namespace oneflow

#endif  // PLATFORM_POSIX
