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
#ifndef ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_HELPER_H_
#define ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_HELPER_H_

#include "oneflow/core/comm_network/epoll/io_event_poller.h"
#include "oneflow/core/comm_network/epoll/socket_read_helper.h"
#include "oneflow/core/comm_network/epoll/socket_write_helper.h"

#ifdef OF_PLATFORM_POSIX

namespace oneflow {

class SocketHelper final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SocketHelper);
  SocketHelper() = delete;
  ~SocketHelper();

  SocketHelper(int sockfd, IOEventPoller* poller);

  void AsyncWrite(const SocketMsg& msg);

 private:
  SocketReadHelper* read_helper_;
  SocketWriteHelper* write_helper_;
};

}  // namespace oneflow

#endif  // OF_PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_SOCKET_HELPER_H_
