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
#ifndef ONEFLOW_CORE_NETWORKER_NETWORKER_MESSAGE_H_
#define ONEFLOW_CORE_NETWORKER_NETWORKER_MESSAGE_H_

#include "oneflow/core/common/platform.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/comm_network/comm_network.h"

#ifdef PLATFORM_POSIX

#include <arpa/inet.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <sys/epoll.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include "oneflow/core/actor/actor_message.h"

namespace oneflow {

enum class NetworkerMsgType {
  kSend,
  kRecv,
  kAck,
};

struct NetworkerMsg {
  uint64_t token;
  void* src_mem_token;
  void* dst_mem_token;
  std::size_t size;
  int64_t dst_machine_id;
  NetworkerMsgType type;
};

}  // namespace oneflow

#endif  // PLATFORM_POSIX

#endif  // ONEFLOW_CORE_NETWORKER_NETWORKER_MESSAGE_H_
