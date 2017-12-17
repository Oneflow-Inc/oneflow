#include "oneflow/core/comm_network/epoll/epoll_comm_network.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/machine_context.h"

#ifdef PLATFORM_POSIX

namespace oneflow {

namespace {

sockaddr_in GetSockAddr(int64_t machine_id, uint16_t port) {
  const Machine& machine = JobDesc::Singleton()->resource().machine(machine_id);
  const std::string& addr = machine.addr();
  sockaddr_in sa;
  sa.sin_family = AF_INET;
  sa.sin_port = htons(port);
  PCHECK(inet_pton(AF_INET, addr.c_str(), &(sa.sin_addr)) == 1);
  return sa;
}

int64_t GetMachineId(const sockaddr_in& sa) {
  char addr[INET_ADDRSTRLEN];
  memset(addr, '\0', sizeof(addr));
  PCHECK(inet_ntop(AF_INET, &(sa.sin_addr), addr, INET_ADDRSTRLEN));
  for (int64_t i = 0; i < JobDesc::Singleton()->TotalMachineNum(); ++i) {
    if (JobDesc::Singleton()->resource().machine(i).addr() == addr) {
      return i;
    }
  }
  UNEXPECTED_RUN();
}

std::string GenPortKey(int64_t machine_id) {
  return "EpollPort/" + std::to_string(machine_id);
}
void PushPort(int64_t machine_id, uint16_t port) {
  CtrlClient::Singleton()->PushKV(GenPortKey(machine_id), std::to_string(port));
}
void ClearPort(int64_t machine_id) {
  CtrlClient::Singleton()->ClearKV(GenPortKey(machine_id));
}
uint16_t PullPort(int64_t machine_id) {
  uint16_t port = 0;
  CtrlClient::Singleton()->PullKV(
      GenPortKey(machine_id),
      [&](const std::string& v) { port = oneflow_cast<uint16_t>(v); });
  return port;
}

}  // namespace

EpollCommNet::~EpollCommNet() {
  for (size_t i = 0; i < pollers_.size(); ++i) {
    LOG(INFO) << "CommNet Thread " << i << " finish";
    pollers_[i]->Stop();
  }
  OF_BARRIER();
  for (IOEventPoller* poller : pollers_) { delete poller; }
  for (auto& pair : sockfd2helper_) { delete pair.second; }
}

void EpollCommNet::Init() {
  CommNet::Singleton()->set_comm_network_ptr(new EpollCommNet());
}

const void* EpollCommNet::RegisterMemory(void* mem_ptr, size_t byte_size) {
  auto mem_desc = new SocketMemDesc;
  mem_desc->mem_ptr = mem_ptr;
  mem_desc->byte_size = byte_size;
  {
    std::unique_lock<std::mutex> lck(mem_desc_mtx_);
    mem_descs_.push_back(mem_desc);
  }
  return mem_desc;
}

void EpollCommNet::UnRegisterMemory(const void* token) {
  std::unique_lock<std::mutex> lck(mem_desc_mtx_);
  CHECK(!mem_descs_.empty());
  unregister_mem_descs_cnt_ += 1;
  if (unregister_mem_descs_cnt_ == mem_descs_.size()) {
    for (SocketMemDesc* mem_desc : mem_descs_) { delete mem_desc; }
    mem_descs_.clear();
    unregister_mem_descs_cnt_ = 0;
  }
}

void EpollCommNet::RegisterMemoryDone() {
  // do nothing
}

void* EpollCommNet::NewActorReadId() { return new ActorReadContext; }

void EpollCommNet::DeleteActorReadId(void* actor_read_id) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  CHECK(actor_read_ctx->read_ctx_list.empty());
  delete actor_read_ctx;
}

void* EpollCommNet::Read(void* actor_read_id, int64_t src_machine_id,
                         const void* src_token, const void* dst_token) {
  // ReadContext
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  ReadContext* read_ctx = new ReadContext;
  read_ctx->done_cnt = 0;
  {
    std::unique_lock<std::mutex> lck(actor_read_ctx->read_ctx_list_mtx);
    actor_read_ctx->read_ctx_list.push_back(read_ctx);
  }
  // request write msg
  SocketMsg msg;
  msg.msg_type = SocketMsgType::kRequestWrite;
  msg.request_write_msg.src_token = src_token;
  msg.request_write_msg.dst_machine_id =
      MachineCtx::Singleton()->this_machine_id();
  msg.request_write_msg.dst_token = dst_token;
  msg.request_write_msg.read_done_id =
      new std::tuple<ActorReadContext*, ReadContext*>(actor_read_ctx, read_ctx);
  GetSocketHelper(src_machine_id)->AsyncWrite(msg);
  return read_ctx;
}

void EpollCommNet::AddReadCallBack(void* actor_read_id, void* read_id,
                                   std::function<void()> callback) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  ReadContext* read_ctx = static_cast<ReadContext*>(read_id);
  if (read_ctx) {
    read_ctx->cbl.push_back(callback);
    return;
  }
  do {
    std::unique_lock<std::mutex> lck(actor_read_ctx->read_ctx_list_mtx);
    if (actor_read_ctx->read_ctx_list.empty()) {
      break;
    } else {
      actor_read_ctx->read_ctx_list.back()->cbl.push_back(callback);
      return;
    }
  } while (0);
  callback();
}

void EpollCommNet::AddReadCallBackDone(void* actor_read_id, void* read_id) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  ReadContext* read_ctx = static_cast<ReadContext*>(read_id);
  if (IncreaseDoneCnt(read_ctx) == 2) {
    FinishOneReadContext(actor_read_ctx, read_ctx);
    delete read_ctx;
  }
}

void EpollCommNet::ReadDone(void* read_done_id) {
  auto parsed_read_done_id =
      static_cast<std::tuple<ActorReadContext*, ReadContext*>*>(read_done_id);
  auto actor_read_ctx = std::get<0>(*parsed_read_done_id);
  auto read_ctx = std::get<1>(*parsed_read_done_id);
  delete parsed_read_done_id;
  if (IncreaseDoneCnt(read_ctx) == 2) {
    {
      std::unique_lock<std::mutex> lck(actor_read_ctx->read_ctx_list_mtx);
      FinishOneReadContext(actor_read_ctx, read_ctx);
    }
    delete read_ctx;
  }
}

int8_t EpollCommNet::IncreaseDoneCnt(ReadContext* read_ctx) {
  std::unique_lock<std::mutex> lck(read_ctx->done_cnt_mtx);
  read_ctx->done_cnt += 1;
  return read_ctx->done_cnt;
}

void EpollCommNet::FinishOneReadContext(ActorReadContext* actor_read_ctx,
                                        ReadContext* read_ctx) {
  CHECK_EQ(actor_read_ctx->read_ctx_list.front(), read_ctx);
  actor_read_ctx->read_ctx_list.pop_front();
  for (std::function<void()>& callback : read_ctx->cbl) { callback(); }
}

void EpollCommNet::SendActorMsg(int64_t dst_machine_id,
                                const ActorMsg& actor_msg) {
  SocketMsg msg;
  msg.msg_type = SocketMsgType::kActor;
  msg.actor_msg = actor_msg;
  GetSocketHelper(dst_machine_id)->AsyncWrite(msg);
}

void EpollCommNet::SendSocketMsg(int64_t dst_machine_id, const SocketMsg& msg) {
  GetSocketHelper(dst_machine_id)->AsyncWrite(msg);
}

EpollCommNet::EpollCommNet() {
  mem_descs_.clear();
  unregister_mem_descs_cnt_ = 0;
  pollers_.resize(JobDesc::Singleton()->CommNetWorkerNum(), nullptr);
  for (size_t i = 0; i < pollers_.size(); ++i) {
    pollers_[i] = new IOEventPoller;
  }
  InitSockets();
  for (IOEventPoller* poller : pollers_) { poller->Start(); }
}

void EpollCommNet::InitSockets() {
  int64_t this_machine_id = MachineCtx::Singleton()->this_machine_id();
  int64_t total_machine_num = JobDesc::Singleton()->TotalMachineNum();
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
  uint16_t this_listen_port = 1024;
  uint16_t listen_port_max = std::numeric_limits<uint16_t>::max();
  for (; this_listen_port < listen_port_max; ++this_listen_port) {
    sockaddr_in this_sockaddr = GetSockAddr(this_machine_id, this_listen_port);
    int bind_result =
        bind(listen_sockfd, reinterpret_cast<sockaddr*>(&this_sockaddr),
             sizeof(this_sockaddr));
    if (bind_result == 0) {
      PCHECK(listen(listen_sockfd, total_machine_num) == 0);
      PushPort(this_machine_id, this_listen_port);
      break;
    } else {
      PCHECK(errno == EACCES || errno == EADDRINUSE);
    }
  }
  CHECK_LT(this_listen_port, listen_port_max);
  // connect
  FOR_RANGE(int64_t, peer_machine_id, this_machine_id + 1, total_machine_num) {
    uint16_t peer_port = PullPort(peer_machine_id);
    sockaddr_in peer_sockaddr = GetSockAddr(peer_machine_id, peer_port);
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    PCHECK(connect(sockfd, reinterpret_cast<sockaddr*>(&peer_sockaddr),
                   sizeof(peer_sockaddr))
           == 0);
    CHECK(sockfd2helper_.emplace(sockfd, NewSocketHelper(sockfd)).second);
    machine_id2sockfd_[peer_machine_id] = sockfd;
  }
  // accept
  FOR_RANGE(int64_t, idx, 0, this_machine_id) {
    sockaddr_in peer_sockaddr;
    socklen_t len = sizeof(peer_sockaddr);
    int sockfd = accept(listen_sockfd,
                        reinterpret_cast<sockaddr*>(&peer_sockaddr), &len);
    PCHECK(sockfd != -1);
    CHECK(sockfd2helper_.emplace(sockfd, NewSocketHelper(sockfd)).second);
    int64_t peer_machine_id = GetMachineId(peer_sockaddr);
    machine_id2sockfd_[peer_machine_id] = sockfd;
  }
  PCHECK(close(listen_sockfd) == 0);
  ClearPort(this_machine_id);
  // useful log
  FOR_RANGE(int64_t, machine_id, 0, total_machine_num) {
    LOG(INFO) << "machine " << machine_id << " sockfd "
              << machine_id2sockfd_[machine_id];
  }
}

SocketHelper* EpollCommNet::GetSocketHelper(int64_t machine_id) {
  int sockfd = machine_id2sockfd_.at(machine_id);
  return sockfd2helper_.at(sockfd);
}

}  // namespace oneflow

#endif  // PLATFORM_POSIX
