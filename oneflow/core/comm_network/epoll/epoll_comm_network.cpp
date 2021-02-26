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
#include "oneflow/core/comm_network/epoll/epoll_comm_network.h"
#include "glog/logging.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/env_desc.h"
#include "oneflow/core/job/global_for.h"

#ifdef OF_PLATFORM_POSIX

#include <netinet/tcp.h>

namespace oneflow {

namespace {

sockaddr_in GetSockAddr(const std::string& addr, uint16_t port) {
  sockaddr_in sa;
  sa.sin_family = AF_INET;
  sa.sin_port = htons(port);
  PCHECK(inet_pton(AF_INET, addr.c_str(), &(sa.sin_addr)) == 1)
      << "addr: " << addr << ", port: " << port;
  return sa;
}

int SockListen(int listen_sockfd, uint16_t listen_port, int32_t total_machine_num) {
  sockaddr_in sa = GetSockAddr("0.0.0.0", listen_port);
  int reuse = 1;
  int ret_setopt =
      setsockopt(listen_sockfd, SOL_SOCKET, SO_REUSEADDR, (const void*)&reuse, sizeof(int));
  CHECK_EQ(ret_setopt, 0);
  int bind_result = bind(listen_sockfd, reinterpret_cast<sockaddr*>(&sa), sizeof(sa));
  if (bind_result == 0) {
    PCHECK(listen(listen_sockfd, total_machine_num) == 0);
    LOG(INFO) << "CommNet:Epoll listening on "
              << "0.0.0.0:" + std::to_string(listen_port);
  } else {
    PCHECK(errno == EACCES || errno == EADDRINUSE) << "SockListen errno: " << errno;
  }
  return bind_result;
}

int64_t GetMachineId(const sockaddr_in& sa) {
  char addr[INET_ADDRSTRLEN];
  memset(addr, '\0', sizeof(addr));
  PCHECK(inet_ntop(AF_INET, &(sa.sin_addr), addr, INET_ADDRSTRLEN));
  for (int64_t i = 0; i < Global<ResourceDesc, ForSession>::Get()->TotalMachineNum(); ++i) {
    if (Global<ResourceDesc, ForSession>::Get()->machine(i).addr() == addr) { return i; }
  }
  UNIMPLEMENTED();
}

std::string GenPortKey(int64_t machine_id) { return "EpollPort/" + std::to_string(machine_id); }
void PushPort(int64_t machine_id, uint16_t port) {
  Global<CtrlClient>::Get()->PushKV(GenPortKey(machine_id), std::to_string(port));
}
void ClearPort(int64_t machine_id) { Global<CtrlClient>::Get()->ClearKV(GenPortKey(machine_id)); }
uint16_t PullPort(int64_t machine_id) {
  uint16_t port = 0;
  Global<CtrlClient>::Get()->PullKV(
      GenPortKey(machine_id), [&](const std::string& v) { port = oneflow_cast<uint16_t>(v); });
  return port;
}

}  // namespace

EpollCommNet::~EpollCommNet() {
  for (size_t i = 0; i < pollers_.size(); ++i) {
    LOG(INFO) << "CommNet Thread " << i << " finish";
    pollers_[i]->Stop();
  }
  // TODO(chengcheng): change to OF_ENV_BARRIER
  OF_SESSION_BARRIER();
  for (IOEventPoller* poller : pollers_) { delete poller; }
  for (auto& pair : sockfd2helper_) { delete pair.second; }
}

void EpollCommNet::RegisterMemoryDone() {
  // do nothing
}

void EpollCommNet::SendActorMsg(int64_t dst_machine_id, const ActorMsg& actor_msg) {
  SocketMsg msg;
  msg.msg_type = SocketMsgType::kActor;
  msg.actor_msg = actor_msg;
  GetSocketHelper(dst_machine_id)->AsyncWrite(msg);
}

void EpollCommNet::SendTransportMsg(int64_t dst_machine_id, const TransportMsg& transport_msg) {
  SocketMsg msg;
  msg.msg_type = SocketMsgType::kTransport;
  msg.transport_msg = transport_msg;
  SendSocketMsg(dst_machine_id, msg);
}

void EpollCommNet::SendSocketMsg(int64_t dst_machine_id, const SocketMsg& msg) {
  GetSocketHelper(dst_machine_id)->AsyncWrite(msg);
}

SocketMemDesc* EpollCommNet::NewMemDesc(void* ptr, size_t byte_size) {
  SocketMemDesc* mem_desc = new SocketMemDesc;
  mem_desc->mem_ptr = ptr;
  mem_desc->byte_size = byte_size;
  return mem_desc;
}

EpollCommNet::EpollCommNet() {
  pollers_.resize(Global<ResourceDesc, ForSession>::Get()->CommNetWorkerNum(), nullptr);
  for (size_t i = 0; i < pollers_.size(); ++i) { pollers_[i] = new IOEventPoller; }
  InitSockets();
  for (IOEventPoller* poller : pollers_) { poller->Start(); }
}

EpollCommNet::EpollCommNet(const Plan& plan) : CommNetIf(plan) {
  pollers_.resize(Global<ResourceDesc, ForSession>::Get()->CommNetWorkerNum(), nullptr);
  for (size_t i = 0; i < pollers_.size(); ++i) { pollers_[i] = new IOEventPoller; }
  InitSockets();
  for (IOEventPoller* poller : pollers_) { poller->Start(); }
}

void EpollCommNet::InitSockets() {
  int64_t this_machine_id = GlobalProcessCtx::Rank();
  auto this_machine = Global<ResourceDesc, ForSession>::Get()->machine(this_machine_id);
  int64_t total_machine_num = Global<ResourceDesc, ForSession>::Get()->TotalMachineNum();
  machine_id2sockfd_.assign(total_machine_num, -1);
  sockfd2helper_.clear();
  size_t poller_idx = 0;
  auto NewSocketHelper = [&](int sockfd) {
    IOEventPoller* poller = pollers_[poller_idx];
    poller_idx = (poller_idx + 1) % pollers_.size();
    return new SocketHelper(sockfd, poller);
  };

  // listen
  int listen_sockfd = socket(AF_INET, SOCK_STREAM, 0);
  int32_t this_listen_port = Global<EnvDesc>::Get()->data_port();
  if (this_listen_port != -1) {
    CHECK_EQ(SockListen(listen_sockfd, this_listen_port, total_machine_num), 0);
    PushPort(this_machine_id,
             ((this_machine.data_port_agent() != -1) ? (this_machine.data_port_agent())
                                                     : (this_listen_port)));
  } else {
    for (this_listen_port = 1024; this_listen_port < GetMaxVal<uint16_t>(); ++this_listen_port) {
      if (SockListen(listen_sockfd, this_listen_port, total_machine_num) == 0) {
        PushPort(this_machine_id, this_listen_port);
        break;
      }
    }
    CHECK_LT(this_listen_port, GetMaxVal<uint16_t>());
  }
  int32_t src_machine_count = 0;

  // connect
  for (int64_t peer_id : peer_machine_id()) {
    if (peer_id < this_machine_id) {
      ++src_machine_count;
      continue;
    }
    uint16_t peer_port = PullPort(peer_id);
    auto peer_machine = Global<ResourceDesc, ForSession>::Get()->machine(peer_id);
    sockaddr_in peer_sockaddr = GetSockAddr(peer_machine.addr(), peer_port);
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    const int val = 1;
    PCHECK(setsockopt(sockfd, IPPROTO_TCP, TCP_NODELAY, (char*)&val, sizeof(int)) == 0);
    PCHECK(connect(sockfd, reinterpret_cast<sockaddr*>(&peer_sockaddr), sizeof(peer_sockaddr))
           == 0);
    CHECK(sockfd2helper_.emplace(sockfd, NewSocketHelper(sockfd)).second);
    machine_id2sockfd_[peer_id] = sockfd;
  }

  // accept
  FOR_RANGE(int32_t, idx, 0, src_machine_count) {
    sockaddr_in peer_sockaddr;
    socklen_t len = sizeof(peer_sockaddr);
    int sockfd = accept(listen_sockfd, reinterpret_cast<sockaddr*>(&peer_sockaddr), &len);
    PCHECK(sockfd != -1);
    CHECK(sockfd2helper_.emplace(sockfd, NewSocketHelper(sockfd)).second);
    int64_t peer_machine_id = GetMachineId(peer_sockaddr);
    machine_id2sockfd_[peer_machine_id] = sockfd;
  }
  PCHECK(close(listen_sockfd) == 0);
  ClearPort(this_machine_id);

  // useful log
  FOR_RANGE(int64_t, machine_id, 0, total_machine_num) {
    LOG(INFO) << "machine " << machine_id << " sockfd " << machine_id2sockfd_[machine_id];
  }
}

SocketHelper* EpollCommNet::GetSocketHelper(int64_t machine_id) {
  int sockfd = machine_id2sockfd_.at(machine_id);
  return sockfd2helper_.at(sockfd);
}

void EpollCommNet::DoRead(void* read_id, int64_t src_machine_id, void* src_token, void* dst_token) {
  SocketMsg msg;
  msg.msg_type = SocketMsgType::kRequestWrite;
  msg.request_write_msg.src_token = src_token;
  msg.request_write_msg.dst_machine_id = GlobalProcessCtx::Rank();
  msg.request_write_msg.dst_token = dst_token;
  msg.request_write_msg.read_id = read_id;
  GetSocketHelper(src_machine_id)->AsyncWrite(msg);
}

}  // namespace oneflow

#endif  // OF_PLATFORM_POSIX
