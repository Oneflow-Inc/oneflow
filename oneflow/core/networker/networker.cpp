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
      case NetworkerMsgType::kPrepareSend: HandlerPrepareSend(msg);
      case NetworkerMsgType::kPrepareRecv: HandlerPrepareRecv(msg);
      case NetworkerMsgType::kSend: HandlerSend(msg);
      case NetworkerMsgType::kAck: HandlerAck(msg);
      default: UNIMPLEMENTED();
    }
  }
}

void Networker::HandlerPrepareSend(const NetworkerMsg& msg) {
  // init status, save callback
  CHECK(token2status_.emplace(msg.token, NetworkerStatus(msg.token)).second);
  NetworkerStatus* stat = &(token2status_.at(msg.token));
  // CHECK(msg.callback != nullptr);
  // stat->callback = msg.callback;
  stat->is_send_ready = true;

  // create src mem token
  CHECK(msg.ptr != nullptr);
  CHECK(msg.size > 0);
  stat->src_mem_token = comm_net_->RegisterMemory(msg.ptr, msg.size);
  stat->src_ptr = msg.ptr;
  stat->size = msg.size;
  stat->dst_machine_id = msg.dst_machine_id;
  stat->src_machine_id = msg.src_machine_id;

  // create kSend msg and send
  SocketMsg socket_msg;
  socket_msg.msg_type = SocketMsgType::kNetworker;
  socket_msg.networker_msg = msg;
  socket_msg.networker_msg.type = NetworkerMsgType::kSend;
  comm_net_->SendSocketMsg(msg.dst_machine_id, socket_msg);
}

void Networker::HandlerPrepareRecv(const NetworkerMsg& msg) {
  // TODO()
}

void Networker::HandlerSend(const NetworkerMsg& msg) {
  // TODO()
}

void Networker::HandlerAck(const NetworkerMsg& msg) {
  // TODO()
}

void Networker::Send(uint64_t token, int64_t dst_machine_id, const void* ptr, std::size_t size,
                     std::function<void()> callback) {
  // Let local networker msg poller prepare send
  NetworkerMsg msg;
  msg.token = token;
  msg.src_machine_id = this_machine_id_;
  msg.dst_machine_id = dst_machine_id;
  msg.ptr = const_cast<void*>(ptr);
  msg.size = size;
  // msg.callback = callback;
  msg.type = NetworkerMsgType::kPrepareSend;
  msg_channel_.Send(msg);
}

void Networker::Receive(uint64_t token, int64_t src_machine_id, void* ptr, std::size_t size,
                        std::function<void()> callback) {
  // Let local networker msg poller prepare recv
  NetworkerMsg msg;
  msg.token = token;
  msg.src_machine_id = src_machine_id;
  msg.dst_machine_id = this_machine_id_;
  msg.ptr = ptr;
  msg.size = size;
  // msg.callback = callback;
  msg.type = NetworkerMsgType::kPrepareRecv;
  msg_channel_.Send(msg);
}

}  // namespace oneflow
