#include "oneflow/core/comm_network/epoll/epoll_comm_network.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/machine_context.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

namespace {

sockaddr_in GetSockAddr(const std::string& addr, uint16_t port) {
  sockaddr_in sa;
  sa.sin_family = AF_INET;
  sa.sin_port = htons(port);
  PCHECK(inet_pton(AF_INET, addr.c_str(), &(sa.sin_addr)) == 1);
  return sa;
}

int32_t SockListen(int32_t listen_sockfd, uint16_t listen_port, int32_t total_machine_num) {
  sockaddr_in sa = GetSockAddr("0.0.0.0", listen_port);
  int32_t bind_result = bind(listen_sockfd, reinterpret_cast<sockaddr*>(&sa), sizeof(sa));
  if (bind_result == 0) {
    PCHECK(listen(listen_sockfd, total_machine_num) == 0);
    LOG(INFO) << "CommNet:Epoll listening on "
              << "0.0.0.0:" + std::to_string(listen_port);
  } else {
    PCHECK(errno == EACCES || errno == EADDRINUSE);
  }
  return bind_result;
}

int64_t GetMachineId(const sockaddr_in& sa) {
  char addr[INET_ADDRSTRLEN];
  memset(addr, '\0', sizeof(addr));
  PCHECK(inet_ntop(AF_INET, &(sa.sin_addr), addr, INET_ADDRSTRLEN));
  for (int64_t i = 0; i < Global<JobDesc>::Get()->TotalMachineNum(); ++i) {
    if (Global<JobDesc>::Get()->resource().machine(i).addr() == addr) { return i; }
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
    pollers_.at(i)->Stop();
  }
  OF_BARRIER();
  for (IOEventPoller* poller : pollers_) { delete poller; }
  for (auto& pair : sockfd2helper_) { delete pair.second; }
}

void EpollCommNet::RegisterMemoryDone() {
  for (void* dst_token : mem_descs()) { dst_token2part_done_cnt_[dst_token] = 0; }
}

void EpollCommNet::SendActorMsg(int64_t dst_machine_id, const ActorMsg& actor_msg) {
  SocketMsg msg;
  msg.msg_type = SocketMsgType::kActor;
  msg.actor_msg = actor_msg;
  GetSocketHelper(dst_machine_id, epoll_conf_.link_num() - 1)->AsyncWrite(msg);
}

void EpollCommNet::RequestRead(int64_t dst_machine_id, void* src_token, void* dst_token,
                               int64_t stream_id) {
  CHECK(src_token);
  CHECK(dst_token);
  int32_t total_byte_size = static_cast<const SocketMemDesc*>(src_token)->byte_size;
  CHECK_GT(total_byte_size, 0);

  int32_t part_length = (total_byte_size + epoll_conf_.link_num() - 1) / epoll_conf_.link_num();
  part_length = RoundUp(part_length, epoll_conf_.msg_segment_kbyte() * 1024);
  int32_t part_num = (total_byte_size + part_length - 1) / part_length;
  CHECK_LE(part_num, epoll_conf_.link_num());

  FOR_RANGE(int32_t, link_index, 0, part_num) {
    int32_t byte_size = (total_byte_size > part_length) ? (part_length) : (total_byte_size);
    CHECK_GT(byte_size, 0);
    total_byte_size -= byte_size;
    CHECK_GE(total_byte_size, 0);
    SocketMsg msg;
    msg.msg_type = SocketMsgType::kRequestRead;
    msg.request_read_msg.src_token = src_token;
    msg.request_read_msg.dst_token = dst_token;
    msg.request_read_msg.offset = link_index * part_length;
    msg.request_read_msg.byte_size = byte_size;
    msg.request_read_msg.stream_id = stream_id;
    msg.request_read_msg.part_num = part_num;
    GetSocketHelper(dst_machine_id, link_index)->AsyncWrite(msg);
  }
  CHECK_EQ(total_byte_size, 0);
}

SocketMemDesc* EpollCommNet::NewMemDesc(void* ptr, size_t byte_size) {
  SocketMemDesc* mem_desc = new SocketMemDesc;
  mem_desc->mem_ptr = ptr;
  mem_desc->byte_size = byte_size;
  return mem_desc;
}

EpollCommNet::EpollCommNet(const Plan& plan)
    : CommNetIf(plan), epoll_conf_(Global<JobDesc>::Get()->epoll_conf()), poller_idx_(0) {
  int64_t poller_num = Global<JobDesc>::Get()->CommNetWorkerNum();
  int64_t peer_mchn_num = peer_machine_id().size();
  if (poller_num < epoll_conf_.link_num()) {
    poller_num = epoll_conf_.link_num();
  } else if (poller_num > (peer_mchn_num * epoll_conf_.link_num())) {
    poller_num = peer_mchn_num * epoll_conf_.link_num();
  }
  for (auto peer_mchn_id : peer_machine_id()) { CHECK(machine_id2sockfds_[peer_mchn_id].empty()); }
  pollers_.resize(poller_num, nullptr);
  for (size_t i = 0; i < pollers_.size(); ++i) { pollers_.at(i) = new IOEventPoller; }
  InitSockets();
  for (IOEventPoller* poller : pollers_) { poller->Start(); }
}

void EpollCommNet::InitSockets() {
  int64_t this_machine_id = Global<MachineCtx>::Get()->this_machine_id();
  auto this_machine = Global<JobDesc>::Get()->resource().machine(this_machine_id);
  int64_t total_machine_num = Global<JobDesc>::Get()->TotalMachineNum();
  sockfd2helper_.clear();

  // listen
  int32_t listen_sockfd = socket(AF_INET, SOCK_STREAM, 0);
  int32_t this_listen_port = Global<JobDesc>::Get()->resource().data_port();
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
  for (int64_t peer_mchn_id : peer_machine_id()) {
    if (peer_mchn_id < this_machine_id) {
      ++src_machine_count;
      continue;
    }
    uint16_t peer_port = PullPort(peer_mchn_id);
    auto peer_machine = Global<JobDesc>::Get()->resource().machine(peer_mchn_id);
    sockaddr_in peer_sockaddr = GetSockAddr(peer_machine.addr(), peer_port);
    FOR_RANGE(int32_t, i, 0, epoll_conf_.link_num()) {
      int32_t sockfd = socket(AF_INET, SOCK_STREAM, 0);
      PCHECK(sockfd != -1);
      PCHECK(connect(sockfd, reinterpret_cast<sockaddr*>(&peer_sockaddr), sizeof(peer_sockaddr))
             == 0);
      SetSocketHelper(peer_mchn_id, sockfd);
    }
  }

  // accept
  FOR_RANGE(int32_t, idx, 0, src_machine_count * epoll_conf_.link_num()) {
    sockaddr_in peer_sockaddr;
    socklen_t len = sizeof(peer_sockaddr);
    int32_t sockfd = accept(listen_sockfd, reinterpret_cast<sockaddr*>(&peer_sockaddr), &len);
    PCHECK(sockfd != -1);
    int64_t peer_mchn_id = GetMachineId(peer_sockaddr);
    SetSocketHelper(peer_mchn_id, sockfd);
  }
  PCHECK(close(listen_sockfd) == 0);
  ClearPort(this_machine_id);

  // useful log
  for (int64_t peer_mchn_id : peer_machine_id()) {
    auto& sockfds = machine_id2sockfds_.at(peer_mchn_id);
    CHECK_EQ(machine_id2sockfds_.at(peer_mchn_id).size(), epoll_conf_.link_num());
    FOR_RANGE(int32_t, link_index, 0, epoll_conf_.link_num()) {
      int32_t sockfd = sockfds.at(link_index);
      CHECK_GT(sockfd, 0);
      LOG(INFO) << "machine: " << peer_mchn_id << ", link index: " << link_index
                << ", sockfd: " << sockfd;
    }
  }
}

void EpollCommNet::SetSocketHelper(int64_t machine_id, int32_t sockfd) {
  poller_idx_ = (poller_idx_) % pollers_.size();
  CHECK(
      sockfd2helper_.emplace(sockfd, new SocketHelper(sockfd, pollers_.at(poller_idx_++))).second);
  machine_id2sockfds_.at(machine_id).push_back(sockfd);
}

SocketHelper* EpollCommNet::GetSocketHelper(int64_t machine_id, int32_t link_index) const {
  int32_t sockfd = machine_id2sockfds_.at(machine_id).at(link_index);
  return sockfd2helper_.at(sockfd);
}

void EpollCommNet::DoRead(int64_t stream_id, int64_t src_machine_id, void* src_token,
                          void* dst_token) {
  SocketMsg msg;
  msg.msg_type = SocketMsgType::kRequestWrite;
  msg.request_write_msg.src_token = src_token;
  msg.request_write_msg.dst_machine_id = Global<MachineCtx>::Get()->this_machine_id();
  msg.request_write_msg.dst_token = dst_token;
  msg.request_write_msg.stream_id = stream_id;
  dst_token2part_done_cnt_.at(dst_token) = 0;
  GetSocketHelper(src_machine_id, epoll_conf_.link_num() - 1)->AsyncWrite(msg);
}

void EpollCommNet::PartReadDone(int64_t stream_id, void* dst_token, int32_t part_num) {
  if (dst_token2part_done_cnt_.at(dst_token).fetch_add(1, std::memory_order_relaxed)
      == (part_num - 1)) {
    ReadDone(stream_id);
  }
}

}  // namespace oneflow

#endif  // PLATFORM_POSIX
