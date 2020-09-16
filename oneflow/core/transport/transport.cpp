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
#include "oneflow/core/transport/transport.h"
#include "oneflow/core/job/machine_context.h"

namespace oneflow {

Transport::Transport() {
  comm_net_ = Global<EpollCommNet>::Get();
  this_machine_id_ = Global<MachineCtx>::Get()->this_machine_id();
  CHECK(comm_net_ != nullptr);
  // maybe need new read id for each dst machine id, maybe need 2 * machine num read ids
  read_id_ = comm_net_->NewActorReadId();
  msg_poller_ = std::thread([this]() { PollMsgChannel(); });
  /*
  callback_poller_ = std::thread([this]() {
    std::function<void()> callback;
    while (callback_channel_.Receive(&callback) == kChannelStatusSuccess) { callback(); }
  });
  */
}

Transport::~Transport() {
  msg_channel_.Close();
  msg_poller_.join();
  // callback_poller_.join();
  CHECK(token2status_.empty());
  // callback_channel_.Close();
  comm_net_->DeleteActorReadId(read_id_);
}

void Transport::EnqueueTransportMsg(const TransportMsg& msg) { msg_channel_.Send(msg); }

void Transport::PollMsgChannel() {
  TransportMsg msg;
  while (msg_channel_.Receive(&msg) == kChannelStatusSuccess) {
    std::cout << " cclog: Oh! I got one message : "
              << "\n ----  token = " << msg.token
              << "\n ----  src_mem_token = " << msg.src_mem_token
              << "\n ----  dst_mem_token = " << msg.dst_mem_token << "\n ----  size = " << msg.size
              << "\n ----  src_machine_id = " << msg.src_machine_id
              << "\n ----  dst_machine_id = " << msg.dst_machine_id
              << "\n ----  type = " << msg.type << std::endl;
    switch (msg.type) {
      case TransportMsgType::kSend: {
        HandlerReceiveSendMsgFromSrcMachine(msg);
        break;
      }
      case TransportMsgType::kAck: {
        HandlerReceiveAckMsgFromDstMachine(msg);
        break;
      }
      default: UNIMPLEMENTED(); break;
    }
  }
}

void Transport::HandlerReceiveSendMsgFromSrcMachine(const TransportMsg& msg) {
  // this handler means that:
  // this machine is dst machine, and receive Send msg from source machine
  CHECK_EQ(msg.type, TransportMsgType::kSend);
  CHECK(msg.src_mem_token != nullptr);
  CHECK(msg.dst_mem_token == nullptr);
  uint64_t token = msg.token;
  CHECK(token != -1);

  // There are two ways to trigger the creation of TransportStatus:
  //   1. the dst machine receives SendMsg from src machine
  //   2. time the dst machine calls the Receive() method.
  // The early party is responsible for creating the TransportStatus,
  // and the late party is responsible for checking the state and calling the DoRead() operation.
  // Early arrival and late arrival are within the protection scope of lock(status_lock_),
  // so there will be no early or late arrival at the same time.

  // prepare transport status for this token.
  // store callback.
  TransportStatus* stat = nullptr;

  // if recv_before_send is ture, it means the Receive() method has been called before this handler
  bool recv_before_send = false;
  {
    std::unique_lock<std::mutex> lock(status_lock_);
    auto it = token2status_.find(token);
    if (it == token2status_.end()) {
      token2status_.emplace(token, TransportStatus(token));
      stat = &(token2status_.at(token));

      // init stat
      // These three members must be initialized in the block protected by lock
      //  to prevent multithreaded access bugs
      stat->size = msg.size;
      stat->src_machine_id = msg.src_machine_id;
      stat->dst_machine_id = msg.dst_machine_id;
    } else {
      recv_before_send = true;
      stat = &(it->second);
    }
  }

  stat->is_send_ready = true;
  CHECK(stat->src_mem_token == nullptr);
  stat->src_mem_token = msg.src_mem_token;

  if (recv_before_send) {
    // it means the local machine has call Transport::Receive() before this handler
    // check status
    CHECK_EQ(stat->size, msg.size);
    CHECK_EQ(stat->src_machine_id, msg.src_machine_id);
    CHECK_EQ(stat->dst_machine_id, msg.dst_machine_id);

    // the recv is ready, and the send is ready too, so call DoRead();
    DoRead(token);
  }
}

void Transport::HandlerReceiveAckMsgFromDstMachine(const TransportMsg& msg) {
  // this handler means that:
  // this machine is src machine, and receive Ack msg from dst machine
  // The Send/Receive is done.
  std::cout << "cclog: Recv ACK msg from dst machine, the src_mem_token is " << msg.src_mem_token
            << std::endl;
  CHECK_EQ(msg.type, TransportMsgType::kAck);
  CHECK(msg.src_mem_token != nullptr);
  CHECK(msg.dst_mem_token != nullptr);
  uint64_t token = msg.token;
  CHECK(token != -1);

  std::function<void()> callback;

  // get status from map
  TransportStatus* stat = nullptr;
  {
    std::unique_lock<std::mutex> lock(status_lock_);
    auto it = token2status_.find(token);
    CHECK(it != token2status_.end());
    stat = &(it->second);

    // check msg == stat
    CHECK_EQ(stat->src_mem_token, msg.src_mem_token);
    CHECK_EQ(stat->size, msg.size);
    CHECK_EQ(stat->src_machine_id, msg.src_machine_id);
    CHECK_EQ(stat->dst_machine_id, msg.dst_machine_id);
    CHECK(stat->callback != nullptr);

    callback = stat->callback;

    // Recovery status
    token2status_.erase(it);
  }

  // Do Send callback
  callback();
}

void Transport::Send(uint64_t token, int64_t dst_machine_id, const void* ptr, std::size_t size,
                     std::function<void()> callback) {
  // prepare transport status for this token.
  // store callback.
  TransportStatus* stat = nullptr;
  {
    std::unique_lock<std::mutex> lock(status_lock_);
    CHECK(token2status_.find(token)
          == token2status_.end());  // this token must be first add to status
    token2status_.emplace(token, TransportStatus(token));
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
  TransportMsg msg;
  msg.token = token;
  msg.src_machine_id = stat->src_machine_id;
  msg.dst_machine_id = stat->dst_machine_id;
  msg.size = size;
  msg.src_mem_token = stat->src_mem_token;
  msg.type = TransportMsgType::kSend;
  comm_net_->SendTransportMsg(msg.dst_machine_id, msg);
}

void Transport::Receive(uint64_t token, int64_t src_machine_id, void* ptr, std::size_t size,
                        std::function<void()> callback) {
  // prepare transport status for this token.
  // store callback.
  TransportStatus* stat = nullptr;

  // if recv_before_send is ture, it means the SendMsg has been handled before this Receive called.
  bool send_before_recv = false;
  {
    std::unique_lock<std::mutex> lock(status_lock_);
    auto it = token2status_.find(token);
    if (it == token2status_.end()) {
      token2status_.emplace(token, TransportStatus(token));
      stat = &(token2status_.at(token));

      // init stat
      // These three members must be initialized in the block protected by lock
      //  to prevent multithreaded access bugs
      stat->size = size;
      stat->src_machine_id = src_machine_id;
      stat->dst_machine_id = this_machine_id_;
    } else {
      send_before_recv = true;
      stat = &(it->second);
    }
  }

  stat->callback = callback;
  stat->is_recv_ready = true;
  CHECK(stat->dst_mem_token == nullptr);
  stat->dst_mem_token = comm_net_->RegisterMemory(ptr, size);

  if (send_before_recv) {
    // it means the source machine has send message to this machine
    // check status
    CHECK_EQ(stat->size, size);
    CHECK_EQ(stat->src_machine_id, src_machine_id);
    CHECK_EQ(stat->dst_machine_id, this_machine_id_);

    // the recv is ready, and the send is ready too, so call DoRead();
    DoRead(token);
  }
}

void Transport::DoRead(uint64_t token) {
  TransportStatus* stat = nullptr;
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
    TransportMsg msg;
    msg.token = stat->token;
    msg.src_machine_id = stat->src_machine_id;
    msg.dst_machine_id = stat->dst_machine_id;
    msg.size = stat->size;
    msg.src_mem_token = stat->src_mem_token;
    msg.dst_mem_token = stat->dst_mem_token;
    msg.type = TransportMsgType::kAck;
    std::cout << "cclog: this_machine_id is " << this_machine_id_ << std::endl;
    std::cout << "cclog: Send ACK msg to src machine, the src_mem_token is " << msg.src_mem_token
              << " dst mem token is " << msg.dst_mem_token << std::endl;
    std::cout << "cclog: Send ACK msg to src machine id is " << msg.src_machine_id
              << " dst machine id is " << msg.dst_machine_id << std::endl;
    comm_net_->SendTransportMsg(msg.src_machine_id, msg);

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
