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
#include "oneflow/core/comm_network/epoll/socket_helper.h"

#ifdef OF_PLATFORM_POSIX

namespace oneflow {

SocketHelper::SocketHelper(int sockfd, IOEventPoller* poller) {
  read_helper_ = new SocketReadHelper(sockfd);
  write_helper_ = new SocketWriteHelper(sockfd, poller);
  poller->AddFd(
      sockfd, [this]() { read_helper_->NotifyMeSocketReadable(); },
      [this]() { write_helper_->NotifyMeSocketWriteable(); });
}

SocketHelper::~SocketHelper() {
  delete read_helper_;
  delete write_helper_;
}

void SocketHelper::AsyncWrite(const SocketMsg& msg) { write_helper_->AsyncWrite(msg); }

}  // namespace oneflow

#endif  // OF_PLATFORM_POSIX
