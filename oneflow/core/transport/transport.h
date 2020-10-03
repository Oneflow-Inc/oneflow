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
#ifndef ONEFLOW_CORE_TRANSPORT_TRANSPORT_H_
#define ONEFLOW_CORE_TRANSPORT_TRANSPORT_H_

#include "oneflow/core/common/channel.h"
#include "oneflow/core/comm_network/epoll/epoll_comm_network.h"
#include "oneflow/core/transport/transport_message.h"

namespace oneflow {

// Transport supports sending and receiving data between two machines, which is identified by
// a unique token.
//
// Suppose machine A wants to send a piece of data to machine B. Global<Transport> both need
// created on machine A and machine B respectively.
//
// Machin A need call:
//   Global<Transport>::Get()->Send(token, B, data_ptr_A, data_size_A, callback_after_send);
// Machin B need call:
//   Global<Transport>::Get()->Receive(token, A, data_ptr_B, data_size_B, callback_after_receive);
//
// data_size_A <= data_size_B
//
// Both call: Send()/Receive() will be executed asynchronously.
//
// When the data transmission is completed, the callbacks of the two machines callback_after_send()
// and callback_after_receive() will be executed on their respective machines.
//
// Transport supports send and receive data on local machine.
//
class Transport {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Transport);
  virtual ~Transport();

  void Send(uint64_t token, int64_t dst_machine_id, const void* ptr, std::size_t size,
            std::function<void()> callback);
  void Receive(uint64_t token, int64_t src_machine_id, void* ptr, std::size_t max_size,
               std::function<void()> callback);
  void EnqueueTransportMsg(const TransportMsg& msg);

 private:
  void PollMsgChannel();
  void HandlerAchievedTransportSendMsgFromSrcMachine(const TransportMsg& msg);
  void HandlerAchievedTransportAckMsgFromDstMachine(const TransportMsg& msg);
  void DoRead(uint64_t token);
  void SendToLocalMachine(uint64_t token, void* ptr, std::size_t size,
                          std::function<void()> callback);
  void RecvFromLocalMachine(uint64_t token, void* ptr, std::size_t max_size,
                            std::function<void()> callback);

  // TODO(chengcheng)
  // Global<Transport> has a dependency on Global<CommNet> which should be initialized first.
  friend class Global<Transport>;
  Transport();

  // TransportStatus stores all the information that Transport needs in a Send / Receive process.
  //
  // At the sender (source machine), the TransportStatus stores the callback from the Send().
  // At the receiver (destination machine), the TransportStatus stores the callback from Receive().
  //
  // In the process of one transmission between two machines, the TransportStatus will be created,
  // changed and finally deleted by sending and receiving messages for many times.
  struct TransportStatus {
    const uint64_t token;
    std::function<void()> callback;
    bool is_send_ready;
    bool is_recv_ready;
    void* src_mem_token;
    void* dst_mem_token;
    // NOTE(chengcheng): must store dst_ptr in status when Receive max_size > Send size
    void* dst_ptr;
    std::size_t size;
    int64_t src_machine_id;
    int64_t dst_machine_id;
    TransportStatus(uint64_t tk)
        : token(tk),
          callback(nullptr),
          is_send_ready(false),
          is_recv_ready(false),
          src_mem_token(nullptr),
          dst_mem_token(nullptr),
          size(-1),
          src_machine_id(-1),
          dst_machine_id(-1) {}
  };

  // CopyStatusOnLocalMachine is a stored state to support local data transfer.
  //
  // This state stores only the most necessary information.
  //
  // When Send() is called first, it stores the token, pointer, size and callback of the sender.
  // In this way, when Receive() is called, copy and two callbacks can be executed.
  //
  // When Receive() is called first, it stores the token, pointer, size and callback of the
  // receiver. In this way, when Send() is called, copy and two callbacks can be executed.
  struct CopyStatusOnLocalMachine {
    const uint64_t token;
    void* ptr;
    std::size_t size;
    std::function<void()> callback;
    CopyStatusOnLocalMachine(uint64_t tk, void* p, std::size_t s, std::function<void()> cb)
        : token(tk), ptr(p), size(s), callback(std::move(cb)) {}
  };

  // Store the TransportStatus for each token (Send/Receive pair).
  // The map token2status_ should be protected by status_mutex_ when you want to change it.
  std::mutex status_mutex_;
  HashMap<uint64_t, TransportStatus> token2status_;

  // for local copy
  std::mutex local_copy_lock_;
  HashMap<uint64_t, CopyStatusOnLocalMachine> token2local_copy_status_;

  int64_t this_machine_id_;
  void* read_id_;
  EpollCommNet* comm_net_;

  Channel<TransportMsg> msg_channel_;
  std::thread msg_poller_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_TRANSPORT_TRANSPORT_H_
