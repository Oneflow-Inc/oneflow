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
#include "oneflow/core/job/machine_context.h"

namespace oneflow {

Networker::Networker() {
  comm_net_ = Global<EpollCommNet>::Get();
  this_machine_id_ = Global<MachineCtx>::Get()->this_machine_id();
  CHECK(comm_net_ != nullptr);
  // maybe need new read id for each dst machine id, maybe need 2 * machine num read ids
  read_id_ = comm_net_->NewActorReadId();
  msg_poller_ = std::thread([this]() { PollMsgChannel(); });
  callback_poller_ = std::thread([this]() {
    std::function<void()> callback;
    while (callback_channel_.Receive(&callback) == kChannelStatusSuccess) { callback(); }
  });
}

Networker::~Networker() {
  msg_poller_.join();
  callback_poller_.join();
  CHECK(token2status_.empty());
  msg_channel_.Close();
  callback_channel_.Close();
  comm_net_->DeleteActorReadId(read_id_);
}

void Networker::EnqueueNetworkerMsg(const NetworkerMsg& msg) { msg_channel_.Send(msg); }

void Networker::PollMsgChannel() {
  NetworkerMsg msg;
  while (msg_channel_.Receive(&msg) == kChannelStatusSuccess) {
    switch (msg.type) {
      case NetworkerMsgType::kSend: HandlerSend(msg);
      case NetworkerMsgType::kAck: HandlerAck(msg);
      default: UNIMPLEMENTED();
    }
  }
}

void Networker::HandlerSend(const NetworkerMsg& msg) {
  // TODO()
}

void Networker::HandlerAck(const NetworkerMsg& msg) {
  // TODO()
}

void Networker::Send(uint64_t token, int64_t dst_machine_id, const void* ptr, std::size_t size,
                     std::function<void()> callback) {
  // prepare networker status for this token.
  // store callback.
  NetworkerStatus* stat = nullptr;
  {
    std::unique_lock<std::mutex> lock(status_lock_);
    CHECK(token2status_.find(token)
          == token2status_.end());  // this token must be first add to status
    token2status_.emplace(token, NetworkerStatus(token));
    stat = &(token2status_.at(token));
  }
  void* mut_ptr = const_cast<void*>(ptr);
  stat->callback = callback;
  stat->is_send_ready = true;
  // stat->is_recv_ready = false;
  stat->src_mem_token = comm_net_->RegisterMemory(mut_ptr, size);
  // stat->dst_mem_token = nullptr;
  stat->src_ptr = mut_ptr;
  // stat->dst_ptr = nullptr;
  stat->size = size;
  stat->src_machine_id = this_machine_id_;
  stat->dst_machine_id = dst_machine_id;

  // Send msg to dst machine
  NetworkerMsg msg;
  msg.token = token;
  msg.src_machine_id = stat->src_machine_id;
  msg.dst_machine_id = stat->dst_machine_id;
  msg.ptr = stat->src_ptr;
  msg.size = size;
  msg.src_mem_token = stat->src_mem_token;
  msg.type = NetworkerMsgType::kSend;
  comm_net_->SendNetworkerMsg(msg.dst_machine_id, msg);
}

void Networker::Receive(uint64_t token, int64_t src_machine_id, void* ptr, std::size_t size,
                        std::function<void()> callback) {
  TODO();
  /*
  // Let local networker msg poller prepare recv
  NetworkerMsg msg;
  msg.token = token;
  msg.src_machine_id = src_machine_id;
  msg.dst_machine_id = this_machine_id_;
  msg.ptr = ptr;
  msg.size = size;
  msg.callback = callback;
  msg.type = NetworkerMsgType::kPrepareRecv;
  msg_channel_.Send(msg);
  */
}

}  // namespace oneflow
