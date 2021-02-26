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
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/transport/transport.h"

namespace oneflow {

Transport::Transport() {
  comm_net_ = Global<EpollCommNet>::Get();
  this_machine_id_ = GlobalProcessCtx::Rank();
  CHECK(comm_net_ != nullptr);
  // maybe need new read id for each dst machine id, maybe need 2 * machine num read ids
  read_id_ = comm_net_->NewActorReadId();
  msg_poller_ = std::thread([this]() { PollMsgChannel(); });
}

Transport::~Transport() {
  msg_channel_.Close();
  msg_poller_.join();
  CHECK(token2status_.empty());
  comm_net_->DeleteActorReadId(read_id_);
}

void Transport::EnqueueTransportMsg(const TransportMsg& msg) {
  CHECK_EQ(msg_channel_.Send(msg), kChannelStatusSuccess);
}

void Transport::PollMsgChannel() {
  TransportMsg msg;
  while (true) {
    ChannelStatus stat = msg_channel_.Receive(&msg);
    if (stat != kChannelStatusSuccess) {
      CHECK_EQ(stat, kChannelStatusErrorClosed);
      break;
    }
    switch (msg.type) {
      case TransportMsgType::kSend: {
        HandlerAchievedTransportSendMsgFromSrcMachine(msg);
        break;
      }
      case TransportMsgType::kAck: {
        HandlerAchievedTransportAckMsgFromDstMachine(msg);
        break;
      }
      default: UNIMPLEMENTED(); break;
    }
  }
}

void Transport::HandlerAchievedTransportSendMsgFromSrcMachine(const TransportMsg& msg) {
  // This machine is dst machine, and receive Send msg from source machine
  // Mayby we need create TransportStatus,
  // or we need update TransportStatus and DoRead().
  CHECK_EQ(msg.type, TransportMsgType::kSend);
  CHECK(msg.src_mem_token != nullptr);
  CHECK(msg.dst_mem_token == nullptr);
  uint64_t token = msg.token;
  CHECK(token != -1);

  // There are two ways to trigger the creation of TransportStatus:
  //   1. The time (T_A) when the dst machine receives SendMsg from src machine
  //   2. The time (T_B) when method Receive() called by the dst machine.
  // Because of T_ A and t_ B are both protected by the lock(status_mutex_), so the creation of
  // TransportStatus will NOT trigger at the same time.
  //
  // T_ A maybe earlier than t_ B, maybe later.
  //
  // In either case, the earlier one is responsible for creating the TransportStatus, and the later
  // one is responsible for checking the TransportStatus and then calling the DoRead() operation.

  // prepare transport status for this token.
  // store callback.
  TransportStatus* stat = nullptr;

  // if recv_before_send is ture, it means the Receive() method has been called before this handler
  bool recv_before_send = false;
  {
    std::unique_lock<std::mutex> lock(status_mutex_);
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
      CHECK_GE(stat->size, msg.size);  // NOTE(chengcheng): Recv size may larger than Send size.
      stat->size = msg.size;           // NOTE(chengcheng): msg.size always is smaller one.
    }

    stat->is_send_ready = true;
    CHECK(stat->src_mem_token == nullptr);
    // src_mem_token MUST init in the block protected by lock
    stat->src_mem_token = msg.src_mem_token;
  }

  if (recv_before_send) {
    // it means the local machine has call Transport::Receive() before this handler
    // check status
    CHECK_EQ(stat->src_machine_id, msg.src_machine_id);
    CHECK_EQ(stat->dst_machine_id, msg.dst_machine_id);

    // the recv is ready, and the send is ready too, so call DoRead();
    DoRead(token);
  }
}

void Transport::HandlerAchievedTransportAckMsgFromDstMachine(const TransportMsg& msg) {
  // This machine is src machine, and receive Ack msg from dst machine. The Send/Receive pair of
  // this token is all done. So we can call callback function and erase TransportStatus.
  CHECK_EQ(msg.type, TransportMsgType::kAck);
  CHECK(msg.src_mem_token != nullptr);
  CHECK(msg.dst_mem_token != nullptr);
  uint64_t token = msg.token;
  CHECK(token != -1);
  std::function<void()> callback;

  // get status from map
  {
    std::unique_lock<std::mutex> lock(status_mutex_);
    auto it = token2status_.find(token);
    CHECK(it != token2status_.end());
    TransportStatus* stat = &(it->second);

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

  // UnRegisterMemory
  comm_net_->UnRegisterMemory(msg.src_mem_token);

  // Do Send callback
  callback();
}

void Transport::Send(uint64_t token, int64_t dst_machine_id, const void* ptr, std::size_t size,
                     std::function<void()> callback) {
  void* mut_ptr = const_cast<void*>(ptr);

  // handler for send to local machine
  if (dst_machine_id == this_machine_id_) {
    SendToLocalMachine(token, mut_ptr, size, callback);
    return;
  }

  // prepare transport status for this token.
  // store callback.
  TransportStatus* stat = nullptr;
  {
    std::unique_lock<std::mutex> lock(status_mutex_);
    CHECK(token2status_.find(token)
          == token2status_.end());  // this token must be first add to status
    token2status_.emplace(token, TransportStatus(token));
    stat = &(token2status_.at(token));
  }
  stat->callback = callback;
  stat->is_send_ready = true;
  stat->is_recv_ready = false;
  stat->src_mem_token = comm_net_->RegisterMemory(mut_ptr, size);
  stat->dst_mem_token = nullptr;
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
  msg.dst_mem_token = stat->dst_mem_token;
  msg.type = TransportMsgType::kSend;
  comm_net_->SendTransportMsg(msg.dst_machine_id, msg);
}

void Transport::Receive(uint64_t token, int64_t src_machine_id, void* ptr, std::size_t max_size,
                        std::function<void()> callback) {
  // handler for receive from local machine
  if (src_machine_id == this_machine_id_) {
    RecvFromLocalMachine(token, ptr, max_size, callback);
    return;
  }

  // prepare transport status for this token.
  // store callback.
  TransportStatus* stat = nullptr;

  // if recv_before_send is ture, it means the SendMsg has been handled before this Receive called.
  bool send_before_recv = false;
  {
    std::unique_lock<std::mutex> lock(status_mutex_);
    auto it = token2status_.find(token);
    if (it == token2status_.end()) {
      token2status_.emplace(token, TransportStatus(token));
      stat = &(token2status_.at(token));

      // init stat
      // These three members must be initialized in the block protected by lock
      //  to prevent multithreaded access bugs
      stat->size = max_size;
      stat->src_machine_id = src_machine_id;
      stat->dst_machine_id = this_machine_id_;
    } else {
      send_before_recv = true;
      stat = &(it->second);
    }

    stat->callback = callback;
    stat->is_recv_ready = true;
    // NOTE(chengcheng): Store dst_ptr so that we can create dst_mem_token in DoRead()
    stat->dst_ptr = ptr;
  }

  if (send_before_recv) {
    // it means the source machine has send message to this machine
    // check status
    CHECK_LE(stat->size,
             max_size);  // NOTE(chengcheng): Receive max_size may larger than Send size.
    CHECK_EQ(stat->src_machine_id, src_machine_id);
    CHECK_EQ(stat->dst_machine_id, this_machine_id_);

    // the recv is ready, and the send is ready too, so call DoRead();
    DoRead(token);
  }
}

void Transport::DoRead(uint64_t token) {
  TransportStatus* stat = nullptr;
  {
    std::unique_lock<std::mutex> lock(status_mutex_);
    auto it = token2status_.find(token);
    CHECK(it != token2status_.end());
    stat = &(it->second);

    // dst_mem_token MUST init in the block protected by lock
    CHECK(stat->dst_mem_token == nullptr);
    // NOTE(chengcheng): ONLY at this time, the stat->size is the real size assigned by Send
    stat->dst_mem_token = comm_net_->RegisterMemory(stat->dst_ptr, stat->size);
  }
  CHECK(stat->is_send_ready && stat->is_recv_ready);
  CHECK(stat->src_mem_token != nullptr);
  CHECK(stat->dst_mem_token != nullptr);
  CHECK(stat->src_machine_id != -1);
  CHECK(stat->dst_machine_id != -1);
  CHECK(stat->size != -1);
  CHECK(stat->callback);
  comm_net_->Read(read_id_, stat->src_machine_id, stat->src_mem_token, stat->dst_mem_token);
  comm_net_->AddReadCallBack(read_id_, [stat, this]() {
    // Send ack message to source machine
    TransportMsg msg;
    msg.token = stat->token;
    msg.src_machine_id = stat->src_machine_id;
    msg.dst_machine_id = stat->dst_machine_id;
    msg.size = stat->size;
    msg.src_mem_token = stat->src_mem_token;
    msg.dst_mem_token = stat->dst_mem_token;
    msg.type = TransportMsgType::kAck;
    comm_net_->SendTransportMsg(msg.src_machine_id, msg);

    // UnRegisterMemory
    comm_net_->UnRegisterMemory(msg.dst_mem_token);

    // Do Recive callback
    stat->callback();

    // Recovery status
    {
      std::unique_lock<std::mutex> lock(status_mutex_);
      auto it = token2status_.find(stat->token);
      CHECK(it != token2status_.end());
      token2status_.erase(it);
    }
  });
}

void Transport::SendToLocalMachine(uint64_t token, void* ptr, std::size_t size,
                                   std::function<void()> callback) {
  bool need_do_copy = false;
  bool need_do_callback = false;
  std::function<void()> receive_callback;
  void* dst_ptr = nullptr;
  {
    std::unique_lock<std::mutex> lock(local_copy_lock_);
    auto it = token2local_copy_status_.find(token);
    if (it == token2local_copy_status_.end()) {
      // init local copy status
      token2local_copy_status_.emplace(token, CopyStatusOnLocalMachine(token, ptr, size, callback));
    } else {
      need_do_callback = true;
      receive_callback = std::move(it->second.callback);

      dst_ptr = it->second.ptr;
      CHECK(size <= it->second.size);  // NOTE(chengcheng): Recv size may larger than Send size.

      if (ptr != dst_ptr) { need_do_copy = true; }

      // erase local copy status
      token2local_copy_status_.erase(it);
    }
  }

  if (need_do_copy) { memcpy(dst_ptr, ptr, size); }

  if (need_do_callback) {
    callback();
    receive_callback();
  }
}

void Transport::RecvFromLocalMachine(uint64_t token, void* ptr, std::size_t max_size,
                                     std::function<void()> callback) {
  bool need_do_copy = false;
  bool need_do_callback = false;
  std::function<void()> send_callback;
  void* src_ptr = nullptr;
  std::size_t size = -1;
  {
    std::unique_lock<std::mutex> lock(local_copy_lock_);
    auto it = token2local_copy_status_.find(token);
    if (it == token2local_copy_status_.end()) {
      // init local copy status
      token2local_copy_status_.emplace(token,
                                       CopyStatusOnLocalMachine(token, ptr, max_size, callback));
    } else {
      need_do_callback = true;
      send_callback = std::move(it->second.callback);

      src_ptr = it->second.ptr;
      size = it->second.size;
      CHECK(max_size >= size);  // NOTE(chengcheng): Recv size may larger than Send size.

      if (ptr != src_ptr) { need_do_copy = true; }

      // erase local copy status
      token2local_copy_status_.erase(it);
    }
  }

  if (need_do_copy) { memcpy(ptr, src_ptr, size); }

  if (need_do_callback) {
    callback();
    send_callback();
  }
}

}  // namespace oneflow
