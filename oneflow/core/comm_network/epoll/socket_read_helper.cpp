#include "oneflow/core/comm_network/epoll/socket_read_helper.h"
#include "oneflow/core/actor/actor_message_bus.h"
#include "oneflow/core/comm_network/epoll/epoll_data_comm_network.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

SocketReadHelper::~SocketReadHelper() {
  // do nothing
}

SocketReadHelper::SocketReadHelper(int sockfd) {
  sockfd_ = sockfd;
  SwitchToMsgHeadReadHandle();
}

void SocketReadHelper::NotifyMeSocketReadable() {
  ReadUntilSocketNotReadable();
}

void SocketReadHelper::SwitchToMsgHeadReadHandle() {
  cur_read_handle_ = &SocketReadHelper::MsgHeadReadHandle;
  read_ptr_ = reinterpret_cast<char*>(&cur_msg_);
  read_size_ = sizeof(cur_msg_);
}

void SocketReadHelper::ReadUntilSocketNotReadable() {
  while ((this->*cur_read_handle_)()) {}
}

bool SocketReadHelper::MsgHeadReadHandle() {
  return DoCurRead(&SocketReadHelper::SetStatusWhenMsgHeadDone);
}

bool SocketReadHelper::MsgBodyReadHandle() {
  return DoCurRead(&SocketReadHelper::SetStatusWhenMsgBodyDone);
}

bool SocketReadHelper::DoCurRead(
    void (SocketReadHelper::*set_cur_read_done)()) {
  ssize_t n = read(sockfd_, read_ptr_, read_size_);
  if (n == read_size_) {
    (this->*set_cur_read_done)();
    return true;
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

void SocketReadHelper::SetStatusWhenMsgHeadDone() {
  switch (cur_msg_.msg_type) {
#define MAKE_ENTRY(x, y) \
  case SocketMsgType::k##x: SetStatusWhen##x##MsgHeadDone(); break;
    OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, SOCKET_MSG_TYPE_SEQ);
#undef MAKE_ENTRY
    default: UNEXPECTED_RUN();
  }
}

void SocketReadHelper::SetStatusWhenMsgBodyDone() {
  if (cur_msg_.msg_type == SocketMsgType::kRequestRead) {
    EpollDataCommNet::Singleton()->ReadDone(
        cur_msg_.request_read_msg.read_done_id);
  }
  SwitchToMsgHeadReadHandle();
}

void SocketReadHelper::SetStatusWhenRequestWriteMsgHeadDone() {
  SocketMsg msg_to_send;
  msg_to_send.msg_type = SocketMsgType::kRequestRead;
  msg_to_send.request_read_msg.src_token = cur_msg_.request_write_msg.src_token;
  msg_to_send.request_read_msg.dst_token = cur_msg_.request_write_msg.dst_token;
  msg_to_send.request_read_msg.read_done_id =
      cur_msg_.request_write_msg.read_done_id;
  EpollDataCommNet::Singleton()->SendSocketMsg(
      cur_msg_.request_write_msg.dst_machine_id, msg_to_send);
  SwitchToMsgHeadReadHandle();
}

void SocketReadHelper::SetStatusWhenRequestReadMsgHeadDone() {
  auto mem_desc =
      static_cast<const SocketMemDesc*>(cur_msg_.request_read_msg.dst_token);
  read_ptr_ = reinterpret_cast<char*>(mem_desc->mem_ptr);
  read_size_ = mem_desc->byte_size;
  cur_read_handle_ = &SocketReadHelper::MsgBodyReadHandle;
}

void SocketReadHelper::SetStatusWhenActorMsgHeadDone() {
  ActorMsgBus::Singleton()->SendMsg(cur_msg_.actor_msg);
  SwitchToMsgHeadReadHandle();
}

}  // namespace oneflow

#endif  // PLATFORM_POSIX
