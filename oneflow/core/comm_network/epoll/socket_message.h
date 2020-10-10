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
#ifndef ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_MESSAGE_H_
#define ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_MESSAGE_H_

#include "oneflow/core/common/platform.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/comm_network/comm_network.h"

#ifdef OF_PLATFORM_POSIX

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/transport/transport_message.h"

namespace oneflow {

#define SOCKET_MSG_TYPE_SEQ                         \
  OF_PP_MAKE_TUPLE_SEQ(RequestWrite, request_write) \
  OF_PP_MAKE_TUPLE_SEQ(RequestRead, request_read)   \
  OF_PP_MAKE_TUPLE_SEQ(Actor, actor)                \
  OF_PP_MAKE_TUPLE_SEQ(Transport, transport)

enum class SocketMsgType {
#define MAKE_ENTRY(x, y) k##x,
  OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, SOCKET_MSG_TYPE_SEQ)
#undef MAKE_ENTRY
};

struct RequestWriteMsg {
  void* src_token;
  int64_t dst_machine_id;
  void* dst_token;
  void* read_id;
};

struct RequestReadMsg {
  void* src_token;
  void* dst_token;
  void* read_id;
};

struct SocketMsg {
  SocketMsgType msg_type;
  union {
#define MAKE_ENTRY(x, y) x##Msg y##_msg;
    OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, SOCKET_MSG_TYPE_SEQ)
#undef MAKE_ENTRY
  };
};

using CallBackList = std::list<std::function<void()>>;

}  // namespace oneflow

#endif  // OF_PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_MESSAGE_H_
