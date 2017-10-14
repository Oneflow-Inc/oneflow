#include "oneflow/core/comm_network/epoll/epoll_data_comm_network.h"
#include "oneflow/core/control/ctrl_client.h"
#include "oneflow/core/job/runtime_context.h"

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

}  // namespace

EpollDataCommNet::~EpollDataCommNet() {
  for (size_t i = 0; i < pollers_.size(); ++i) {
    LOG(INFO) << "IOWorker " << i << " finish";
    pollers_[i]->Stop();
  }
  OF_BARRIER();
  for (IOEventPoller* poller : pollers_) { delete poller; }
  for (auto& pair : sockfd2helper_) { delete pair.second; }
}

void EpollDataCommNet::Init() {
  DataCommNet::Singleton()->set_comm_network_ptr(new EpollDataCommNet());
}

const void* EpollDataCommNet::RegisterMemory(void* mem_ptr, size_t byte_size) {
  auto mem_desc = new SocketMemDesc;
  mem_desc->mem_ptr = mem_ptr;
  mem_desc->byte_size = byte_size;
  {
    std::unique_lock<std::mutex> lck(mem_desc_mtx_);
    mem_descs_.push_back(mem_desc);
  }
  return mem_desc;
}

void EpollDataCommNet::UnRegisterMemory(const void* token) {
  std::unique_lock<std::mutex> lck(mem_desc_mtx_);
  CHECK(!mem_descs_.empty());
  unregister_mem_descs_cnt_ += 1;
  if (unregister_mem_descs_cnt_ == mem_descs_.size()) {
    for (SocketMemDesc* mem_desc : mem_descs_) { delete mem_desc; }
    mem_descs_.clear();
    unregister_mem_descs_cnt_ = 0;
  }
}

void EpollDataCommNet::RegisterMemoryDone() {
  // do nothing
}

void* EpollDataCommNet::NewActorReadId() { return new ActorReadContext; }

void EpollDataCommNet::DeleteActorReadId(void* actor_read_id) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  CHECK(actor_read_ctx->read_ctx_list.empty());
  delete actor_read_ctx;
}

void* EpollDataCommNet::Read(void* actor_read_id, int64_t src_machine_id,
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
      RuntimeCtx::Singleton()->this_machine_id();
  msg.request_write_msg.dst_token = dst_token;
  msg.request_write_msg.read_done_id =
      new std::tuple<ActorReadContext*, ReadContext*>(actor_read_ctx, read_ctx);
  GetSocketHelper(src_machine_id)->AsyncWrite(msg);
  return read_ctx;
}

void EpollDataCommNet::AddReadCallBack(void* actor_read_id, void* read_id,
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
    }
  } while (0);
  callback();
}

void EpollDataCommNet::AddReadCallBackDone(void* actor_read_id, void* read_id) {
  auto actor_read_ctx = static_cast<ActorReadContext*>(actor_read_id);
  ReadContext* read_ctx = static_cast<ReadContext*>(read_id);
  IncreaseDoneCnt(actor_read_ctx, read_ctx);
}

void EpollDataCommNet::ReadDone(void* read_done_id) {
  auto parsed_read_done_id =
      static_cast<std::tuple<ActorReadContext*, ReadContext*>*>(read_done_id);
  auto actor_read_ctx = std::get<0>(*parsed_read_done_id);
  auto read_ctx = std::get<1>(*parsed_read_done_id);
  delete parsed_read_done_id;
  IncreaseDoneCnt(actor_read_ctx, read_ctx);
}

void EpollDataCommNet::IncreaseDoneCnt(ActorReadContext* actor_read_ctx,
                                       ReadContext* read_ctx) {
  do {
    std::unique_lock<std::mutex> lck(read_ctx->done_cnt_mtx);
    read_ctx->done_cnt += 1;
    if (read_ctx->done_cnt == 2) {
      break;
    } else {
      return;
    }
  } while (0);
  {
    std::unique_lock<std::mutex> lck(actor_read_ctx->read_ctx_list_mtx);
    CHECK_EQ(actor_read_ctx->read_ctx_list.front(), read_ctx);
    actor_read_ctx->read_ctx_list.pop_front();
    for (std::function<void()>& callback : read_ctx->cbl) { callback(); }
  }
  delete read_ctx;
}

void EpollDataCommNet::SendActorMsg(int64_t dst_machine_id,
                                    const ActorMsg& actor_msg) {
  SocketMsg msg;
  msg.msg_type = SocketMsgType::kActor;
  msg.actor_msg = actor_msg;
  GetSocketHelper(dst_machine_id)->AsyncWrite(msg);
}

void EpollDataCommNet::SendSocketMsg(int64_t dst_machine_id,
                                     const SocketMsg& msg) {
  GetSocketHelper(dst_machine_id)->AsyncWrite(msg);
}

EpollDataCommNet::EpollDataCommNet() {
  mem_descs_.clear();
  unregister_mem_descs_cnt_ = 0;
  pollers_.resize(JobDesc::Singleton()->CommNetIOWorkerNum(), nullptr);
  for (size_t i = 0; i < pollers_.size(); ++i) {
    pollers_[i] = new IOEventPoller;
  }
  InitSockets();
  for (IOEventPoller* poller : pollers_) { poller->Start(); }
}

void EpollDataCommNet::InitSockets() {
  int64_t this_machine_id = RuntimeCtx::Singleton()->this_machine_id();
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
      CtrlClient::Singleton()->PushPort(this_listen_port);
      break;
    } else {
      PCHECK(errno == EACCES || errno == EADDRINUSE);
    }
  }
  CHECK_LT(this_listen_port, listen_port_max);
  // connect
  FOR_RANGE(int64_t, peer_machine_id, this_machine_id + 1, total_machine_num) {
    uint16_t peer_port = CtrlClient::Singleton()->PullPort(peer_machine_id);
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
  // useful log
  FOR_RANGE(int64_t, machine_id, 0, total_machine_num) {
    LOG(INFO) << "machine " << machine_id << " sockfd "
              << machine_id2sockfd_[machine_id];
  }
}

SocketHelper* EpollDataCommNet::GetSocketHelper(int64_t machine_id) {
  int sockfd = machine_id2sockfd_.at(machine_id);
  return sockfd2helper_.at(sockfd);
}

}  // namespace oneflow

#endif  // PLATFORM_POSIX
