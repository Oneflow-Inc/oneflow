/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_READ_HELPER_H_
#define ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_READ_HELPER_H_

#include "oneflow/core/comm_network/epoll/socket_message.h"

#ifdef OF_PLATFORM_POSIX

namespace oneflow {

class SocketReadHelper final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SocketReadHelper);
  SocketReadHelper() = delete;
  ~SocketReadHelper();

  SocketReadHelper(int sockfd);

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

  int sockfd_;

  SocketMsg cur_msg_;
  bool (SocketReadHelper::*cur_read_handle_)();
  char* read_ptr_;
  size_t read_size_;
};

}  // namespace oneflow

#endif  // OF_PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_READ_HELPER_H_
