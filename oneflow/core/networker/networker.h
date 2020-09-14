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
#ifndef ONEFLOW_CORE_NETWORKER_NETWORKER_H_
#define ONEFLOW_CORE_NETWORKER_NETWORKER_H_

#include "oneflow/core/common/channel.h"
#include "oneflow/core/comm_network/epoll/epoll_comm_network.h"
#include "oneflow/core/networker/networker_message.h"

namespace oneflow {

struct NetworkerStatus {
  const uint64_t token;
  std::function<void()> callback;
  bool is_send_ready;
  bool is_recv_ready;
  void* src_mem_token;
  void* dst_mem_token;
  std::size_t size;
  int64_t src_machine_id;
  int64_t dst_machine_id;
  NetworkerStatus(uint64_t tk)
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

class Networker {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Networker);
  virtual ~Networker();

  void Send(uint64_t token, int64_t dst_machine_id, const void* ptr, std::size_t size,
            std::function<void()> callback);
  void Receive(uint64_t token, int64_t src_machine_id, void* ptr, std::size_t size,
               std::function<void()> callback);
  void EnqueueNetworkerMsg(const NetworkerMsg& msg);

 private:
  void PollMsgChannel();
  void HandlerReceiveSendMsgFromSrcMachine(const NetworkerMsg& msg);
  void HandlerReceiveAckMsgFromDstMachine(const NetworkerMsg& msg);
  void DoRead(uint64_t token);

  friend class Global<Networker>;
  Networker();
  std::mutex status_lock_;
  HashMap<uint64_t, NetworkerStatus> token2status_;

  int64_t this_machine_id_;
  void* read_id_;
  EpollCommNet* comm_net_;

  Channel<NetworkerMsg> msg_channel_;
  std::thread msg_poller_;

  // unused now
  Channel<std::function<void()>> callback_channel_;
  std::thread callback_poller_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NETWORKER_NETWORKER_H_
