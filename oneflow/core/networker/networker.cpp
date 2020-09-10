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
#include "oneflow/core/networker/networker.h"
#include "oneflow/core/comm_network/epoll/epoll_comm_network.h"

namespace oneflow {

Networker::Networker() {
  msg_poller_ = std::thread([this]() { PollMsgChannel(); });
}

Networker::~Networker() {
  msg_poller_.join();
  CHECK(token2status_.empty());
  msg_channel_.Close();
}

void Networker::EnqueueNetworkerMsg(const NetworkerMsg& msg) { msg_channel_.Send(msg); }

void Networker::PollMsgChannel() {
  while (true) {
    // TODO
  }
}

void Networker::Send(uint64_t token, int64_t dst_machine_id, const void* ptr, std::size_t size,
                     std::function<void()> callback) {
  // TODO
}

void Networker::Receive(uint64_t token, int64_t src_machine_id, void* ptr, std::size_t size,
                        std::function<void()> callback) {
  // TODO
}

}  // namespace oneflow
