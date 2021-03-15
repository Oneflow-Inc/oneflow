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
#ifndef ONEFLOW_CORE_COMM_NETWORK_EPOLL_EPOLL_COMM_NETWORK_H_
#define ONEFLOW_CORE_COMM_NETWORK_EPOLL_EPOLL_COMM_NETWORK_H_

#ifdef __linux__

#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/comm_network/epoll/socket_helper.h"
#include "oneflow/core/comm_network/epoll/socket_memory_desc.h"

namespace oneflow {

class EpollCommNet final : public CommNetIf<SocketMemDesc> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(EpollCommNet);
  ~EpollCommNet();

  DEPRECATED static void Init(const Plan& plan) {
    Global<CommNet>::SetAllocated(new EpollCommNet(plan));
  }

  void RegisterMemoryDone() override;

  void SendActorMsg(int64_t dst_machine_id, const ActorMsg& msg) override;
  void SendSocketMsg(int64_t dst_machine_id, const SocketMsg& msg);
  void SendTransportMsg(int64_t dst_machine_id, const TransportMsg& msg);

 private:
  SocketMemDesc* NewMemDesc(void* ptr, size_t byte_size) override;

  friend class Global<EpollCommNet>;
  EpollCommNet();
  DEPRECATED EpollCommNet(const Plan& plan);
  void InitSockets();
  SocketHelper* GetSocketHelper(int64_t machine_id);
  void DoRead(void* read_id, int64_t src_machine_id, void* src_token, void* dst_token) override;

  std::vector<IOEventPoller*> pollers_;
  std::vector<int> machine_id2sockfd_;
  HashMap<int, SocketHelper*> sockfd2helper_;
};

}  // namespace oneflow

#endif  // OF_PLATFORM_POSIX

#endif  // ONEFLOW_CORE_COMM_NETWORK_EPOLL_EPOLL_COMM_NETWORK_H_
