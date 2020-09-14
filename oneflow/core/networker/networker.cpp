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
      case NetworkerMsgType::kSend: HandlerReceiveSendMsgFromSrcMachine(msg);
      case NetworkerMsgType::kAck: HandlerReceiveAckMsgFromDstMachine(msg);
      default: UNIMPLEMENTED();
    }
  }
}

void Networker::HandlerReceiveSendMsgFromSrcMachine(const NetworkerMsg& msg) {
  // this handler means that:
  // this machine is dst machine, and receive Send msg from source machine
  CHECK(msg.src_mem_token != nullptr);
  CHECK(msg.dst_mem_token == nullptr);
  uint64_t token = msg.token;
  CHECK(token != -1);

  // prepare networker status for this token.
  // store callback.
  NetworkerStatus* stat = nullptr;
  bool is_recv_ready = false;
  {
    std::unique_lock<std::mutex> lock(status_lock_);
    auto it = token2status_.find(token);
    if (it == token2status_.end()) {
      token2status_.emplace(token, NetworkerStatus(token));
      stat = &(token2status_.at(token));
    } else {
      is_recv_ready = true;
      stat = &(it->second);
    }
  }

  stat->is_send_ready = true;
  CHECK(stat->src_mem_token == nullptr);
  stat->src_mem_token = msg.src_mem_token;

  if (is_recv_ready) {
    // it means the local machine has call Networker::Receive() before this handler
    // check status
    CHECK_EQ(stat->size, msg.size);
    CHECK_EQ(stat->src_machine_id, msg.src_machine_id);
    CHECK_EQ(stat->dst_machine_id, msg.dst_machine_id);
    DoRead(token);
  } else {
    // init and wait for message from source machine
    CHECK(stat->callback == nullptr);
    CHECK(stat->is_recv_ready == false);
    CHECK(stat->dst_mem_token == nullptr);
    stat->size = msg.size;
    stat->src_machine_id = msg.src_machine_id;
    stat->dst_machine_id = msg.dst_machine_id;
  }
}

void Networker::HandlerReceiveAckMsgFromDstMachine(const NetworkerMsg& msg) {
  // this handler means that:
  // this machine is src machine, and receive Ack msg from dst machine
  // The Send/Receive is done.
  std::cout << "cclog: Recv ACK msg from dst machine, the src_mem_token is " << msg.src_mem_token
            << std::endl;
  CHECK(msg.src_mem_token != nullptr);
  CHECK(msg.dst_mem_token != nullptr);
  uint64_t token = msg.token;
  CHECK(token != -1);

  // get status from map
  NetworkerStatus* stat = nullptr;
  {
    std::unique_lock<std::mutex> lock(status_lock_);
    auto it = token2status_.find(token);
    CHECK(it != token2status_.end());
    stat = &(it->second);
  }

  // check msg == stat
  CHECK_EQ(stat->src_mem_token, msg.src_mem_token);
  CHECK_EQ(stat->size, msg.size);
  CHECK_EQ(stat->src_machine_id, msg.src_machine_id);
  CHECK_EQ(stat->dst_machine_id, msg.dst_machine_id);
  CHECK(stat->callback != nullptr);

  // Do Send callback
  stat->callback();

  // Recovery status
  {
    std::unique_lock<std::mutex> lock(status_lock_);
    auto it = token2status_.find(token);
    CHECK(it != token2status_.end());
    token2status_.erase(it);
  }
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
  stat->size = size;
  stat->src_machine_id = this_machine_id_;
  stat->dst_machine_id = dst_machine_id;

  // Send msg to dst machine
  NetworkerMsg msg;
  msg.token = token;
  msg.src_machine_id = stat->src_machine_id;
  msg.dst_machine_id = stat->dst_machine_id;
  msg.size = size;
  msg.src_mem_token = stat->src_mem_token;
  msg.type = NetworkerMsgType::kSend;
  comm_net_->SendNetworkerMsg(msg.dst_machine_id, msg);
}

void Networker::Receive(uint64_t token, int64_t src_machine_id, void* ptr, std::size_t size,
                        std::function<void()> callback) {
  // prepare networker status for this token.
  // store callback.
  NetworkerStatus* stat = nullptr;
  bool is_send_ready = false;
  {
    std::unique_lock<std::mutex> lock(status_lock_);
    auto it = token2status_.find(token);
    if (it == token2status_.end()) {
      token2status_.emplace(token, NetworkerStatus(token));
      stat = &(token2status_.at(token));
    } else {
      is_send_ready = true;
      stat = &(it->second);
    }
  }

  stat->callback = callback;
  stat->is_recv_ready = true;
  CHECK(stat->dst_mem_token == nullptr);
  stat->dst_mem_token = comm_net_->RegisterMemory(ptr, size);

  if (is_send_ready) {
    // it means the source machine has send message to this machine
    // check status
    CHECK(stat->is_send_ready);
    CHECK_EQ(stat->size, size);
    CHECK_EQ(stat->src_machine_id, src_machine_id);
    CHECK_EQ(stat->dst_machine_id, this_machine_id_);
    DoRead(token);
  } else {
    // init and wait for message from source machine
    stat->size = size;
    stat->src_machine_id = src_machine_id;
    stat->dst_machine_id = this_machine_id_;
  }
}

void Networker::DoRead(uint64_t token) {
  NetworkerStatus* stat = nullptr;
  {
    std::unique_lock<std::mutex> lock(status_lock_);
    auto it = token2status_.find(token);
    CHECK(it != token2status_.end());
    stat = &(it->second);
  }
  CHECK(stat != nullptr);
  CHECK(stat->is_send_ready && stat->is_recv_ready);
  CHECK(stat->src_mem_token != nullptr);
  CHECK(stat->dst_mem_token != nullptr);
  CHECK(stat->src_machine_id != -1);
  CHECK(stat->dst_machine_id != -1);
  CHECK(stat->size != -1);
  CHECK(stat->callback != nullptr);
  comm_net_->Read(read_id_, stat->src_machine_id, stat->src_mem_token, stat->dst_mem_token);
  comm_net_->AddReadCallBack(read_id_, [stat, this]() {
    CHECK(stat != nullptr);

    // Send ack message to source machine
    NetworkerMsg msg;
    msg.token = stat->token;
    msg.src_machine_id = stat->src_machine_id;
    msg.dst_machine_id = stat->dst_machine_id;
    msg.size = stat->size;
    msg.src_mem_token = stat->src_mem_token;
    msg.dst_mem_token = stat->dst_mem_token;
    msg.type = NetworkerMsgType::kAck;
    std::cout << "cclog: Send ACK msg to src machine, the src_mem_token is " << msg.src_mem_token
              << std::endl;
    std::cout << "cclog: Send ACK msg to src machine id is " << msg.src_machine_id << std::endl;
    comm_net_->SendNetworkerMsg(msg.src_machine_id, msg);

    // Do Recive callback
    stat->callback();

    // Recovery status
    {
      std::unique_lock<std::mutex> lock(status_lock_);
      auto it = token2status_.find(stat->token);
      CHECK(it != token2status_.end());
      token2status_.erase(it);
    }
  });
}

}  // namespace oneflow
