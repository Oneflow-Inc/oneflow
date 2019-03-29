#ifndef ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_READ_HELPER_H_
#define ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_READ_HELPER_H_

#include "oneflow/core/comm_network/epoll/socket_message.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

class SocketReadHelper final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SocketReadHelper);
  SocketReadHelper() = delete;
  ~SocketReadHelper();

  SocketReadHelper(int32_t sockfd);

  void NotifyMeSocketReadable();

 private:
  void SwitchToMsgHeadReadHandle();
  void ReadUntilSocketNotReadable();

  bool MsgHeadReadHandle();
  bool MsgBodyReadHandle();

  bool DoCurRead(void (SocketReadHelper::*set_cur_read_done)());
  void SetStatusWhenMsgHeadDone();
  void SetStatusWhenMsgBodyDone();

#define MAKE_ENTRY(x, y) void SetStatusWhen##x##MsgHeadDone();
  OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, SOCKET_MSG_TYPE_SEQ);
#undef MAKE_ENTRY

  int32_t sockfd_;

  SocketMsg cur_msg_;
  bool (SocketReadHelper::*cur_read_handle_)();
  char* read_ptr_;
  size_t read_size_;
};

}  // namespace oneflow

#endif  // PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_READ_HELPER_H_
